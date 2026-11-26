"""Passive memory monitoring and between-process reset logic (stdlib only).

Reset model. Every arm runs in its own OS process; device (TPU HBM) memory belongs to
that process and is released when it exits, which is what an interactive kernel restart
achieves. `os._exit(0)` in the notebook kernel is NOT used: in a background (commit)
run the notebook is executed by that single kernel, so killing it ends the whole run.
Instead the driver samples memory passively, kills leftover processes after each arm,
and performs one full reset (fresh venv + reinstall) when memory is above the limits
before the next arm starts, or when an arm finds the accelerator already occupied.
"""
import json, os, shutil, signal, subprocess, threading, time
from pathlib import Path

HBM_DIRTY_EXIT = 75          # pipeline exit code: accelerator memory already occupied at start

DEFAULTS = dict(
    sample_sec=15,            # passive sampling period
    host_used_reset_pct=40.0, # between arms: host RAM used above this -> full reset
    host_used_kill_pct=95.0,  # during an arm: above this for kill_samples samples -> stop that arm
    kill_samples=3,
    disk_used_reset_pct=85.0, # between arms: any watched disk above this -> reset + drop tensor cache
    hbm_dirty_frac=0.10,      # at arm start: bytes_in_use / bytes_limit above this -> dirty accelerator
    max_resets=3,
    settle_sec=10,            # wait after an arm before judging the idle state
)


def config(overrides=None):
    cfg = dict(DEFAULTS)
    unknown = set(overrides or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown memory_guard keys: {sorted(unknown)}')
    cfg.update(overrides or {})
    return cfg


def host_memory(meminfo='/proc/meminfo'):
    """Host RAM from MemAvailable (page cache counts as available). None if unreadable."""
    try:
        kv = {}
        for line in Path(meminfo).read_text().splitlines():
            name, _, rest = line.partition(':')
            kv[name] = int(rest.split()[0]) * 1024
        total, avail = kv['MemTotal'], kv['MemAvailable']
        return dict(total=total, available=avail, used_pct=round(100.0 * (total - avail) / total, 2),
                    shmem=kv.get('Shmem', 0))
    except (OSError, KeyError, ValueError, IndexError):
        return None


def disk_usage(paths):
    out = {}
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            out[str(p)] = round(100.0 * u.used / u.total, 2)
        except OSError:
            pass
    return out


def tpu_holders(proc_root='/proc', exclude=()):
    """PIDs holding an accelerator device node or the libtpu lock file (Linux only)."""
    found = {}
    root = Path(proc_root)
    if not root.is_dir():
        return found
    me = {os.getpid(), *exclude}
    for d in root.iterdir():
        if not d.name.isdigit() or int(d.name) in me:
            continue
        try:
            for fd in (d / 'fd').iterdir():
                target = os.readlink(fd)
                if target.startswith(('/dev/accel', '/dev/vfio/')) or target == '/tmp/libtpu_lockfile':
                    try:
                        cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')[:200]
                    except OSError:
                        cmd = ''
                    found[int(d.name)] = dict(target=target, cmd=cmd)
                    break
        except OSError:
            continue
    return found


def kill_tree(proc, grace=20):
    """Terminate a child started with start_new_session=True together with its group."""
    posix = hasattr(os, 'killpg')

    def send(sig):
        try:
            if posix:
                os.killpg(proc.pid, sig)
            elif proc.poll() is None:
                proc.kill() if sig != signal.SIGTERM else proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    send(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    send(getattr(signal, 'SIGKILL', signal.SIGTERM))
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def idle_check(cfg, disks, holders_fn=tpu_holders, memory_fn=host_memory, disk_fn=disk_usage):
    """State between arms. Returns (reasons, snapshot); non-empty reasons => reset."""
    mem, disk, holders = memory_fn(), disk_fn(disks), holders_fn()
    reasons = []
    if mem and mem['used_pct'] >= cfg['host_used_reset_pct']:
        reasons.append(f"host_ram_used {mem['used_pct']}% >= {cfg['host_used_reset_pct']}%")
    for p, pct in disk.items():
        if pct >= cfg['disk_used_reset_pct']:
            reasons.append(f"disk_used {p} {pct}% >= {cfg['disk_used_reset_pct']}%")
    if holders:
        reasons.append(f'leftover accelerator holders: {sorted(holders)}')
    return reasons, dict(host=mem, disk=disk, holders={str(k): v for k, v in holders.items()})


def clear_holders(log=print, holders_fn=tpu_holders, lockfile='/tmp/libtpu_lockfile'):
    """Kill leftover accelerator holders; remove the libtpu lock only when nobody holds it."""
    killed = []
    for pid in holders_fn():
        try:
            os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM)); killed.append(pid)
        except OSError as e:
            log(f'[mem-guard] could not kill {pid}: {type(e).__name__}')
    if killed:
        time.sleep(3)
    removed = False
    if not holders_fn():
        try:
            Path(lockfile).unlink(); removed = True
        except OSError:
            pass
    return dict(killed=killed, lockfile_removed=removed)


class Monitor:
    """Passive host sampler: appends JSONL, keeps peaks, and asks the driver to stop the
    current arm only on sustained near-exhaustion (so the OS OOM killer never picks the
    driver or the notebook kernel)."""

    def __init__(self, cfg, path, disks, on_overflow=None, memory_fn=host_memory, disk_fn=disk_usage):
        self.cfg, self.path, self.disks, self.on_overflow = cfg, Path(path), disks, on_overflow
        self.memory_fn, self.disk_fn = memory_fn, disk_fn
        self.label, self.over, self.peak = 'setup', 0, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        mem = self.memory_fn()
        row = dict(t=round(time.time(), 1), arm=self.label, host=mem, disk=self.disk_fn(self.disks))
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        if mem:
            self.peak = max(self.peak, mem['used_pct'])
            self.over = self.over + 1 if mem['used_pct'] >= self.cfg['host_used_kill_pct'] else 0
            if self.over >= self.cfg['kill_samples'] and self.on_overflow:
                self.over = 0
                self.on_overflow(f"host RAM {mem['used_pct']}% for {self.cfg['kill_samples']} samples")
        return row

    def _run(self):
        while not self._stop.wait(self.cfg['sample_sec']):
            try:
                self.sample()
            except Exception as e:                      # monitoring must never stop a run
                print('[mem-guard] sample failed:', type(e).__name__, flush=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()


def hbm_stats(devices):
    """Per-device allocator statistics; None entries where the backend reports nothing."""
    out = []
    for d in devices:
        try:
            s = d.memory_stats()
        except Exception:
            s = None
        if s:
            out.append({k: int(s[k]) for k in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit') if k in s})
        else:
            out.append(None)
    return out


def hbm_dirty(stats, frac):
    """Devices whose memory is already occupied beyond `frac` of the limit."""
    bad = []
    for i, s in enumerate(stats):
        if s and s.get('bytes_limit') and s.get('bytes_in_use', 0) / s['bytes_limit'] > frac:
            bad.append(i)
    return bad


def start_hbm_monitor(devices, path, every=30.0):
    """In-process passive accelerator sampler (daemon thread)."""
    stop = threading.Event()

    def run():
        while not stop.wait(every):
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(t=round(time.time(), 1), devices=hbm_stats(devices))) + '\n')
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
    return stop
