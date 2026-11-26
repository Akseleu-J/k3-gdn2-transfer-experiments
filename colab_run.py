"""Sequential tiny/real K3 -> GDN-2/MLA transfer diagnostics with one atomic results ZIP.

Runs under a plain Colab Python with stdlib only: creates an isolated venv
(bootstrap without ensurepip), installs pinned dependencies, executes
experiment.py once per selected phase and keeps the results archive refreshed
every 60 seconds. Adapted from the reviewed single-run flow of the historical
notebooks (KAGGLE_PROTOCOL.md forbids re-submitting those embedded-payload
versions; this file is the readable replacement).
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, threading, time, traceback, uuid, zipfile
from pathlib import Path

PACKAGES = ['jax[tpu]==0.11.2', 'flax==0.12.9', 'optax==0.2.6', 'requests',
            'huggingface_hub', 'transformers>=4.56,<5', 'tiktoken', 'sentencepiece', 'safetensors']


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=('all', 'tiny', 'real', 'staged'), default='tiny')
    ap.add_argument('--script', default='experiment.py', help='repo script to execute per phase (staged pipeline: staged_transfer.py)')
    ap.add_argument('--tiny-steps', type=int, default=30)
    ap.add_argument('--real-steps', type=int, default=0)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--real-batch', type=int, default=0)
    ap.add_argument('--run-root', default='/content/k3_runs')
    ap.add_argument('--venv-root', default='/content')
    ap.add_argument('--drive-base', default=None, help='Directory for periodic archive copies; mounted by the notebook launcher')
    ap.add_argument('--k3-revision', default='main', help='Pinned moonshotai/Kimi-K3 source revision')
    ap.add_argument('--repo-sha', default='', help='Full commit SHA of this repository, recorded for provenance')
    ap.add_argument('--no-venv', action='store_true', help='Local/CI testing: use the current interpreter, skip venv + pip install')
    ap.add_argument('--arms', default=None, help='Repo-relative JSON with a list of arms (name, env, timeout_sec); each arm runs '
                    'as an isolated phase with its own artifacts dir; a failed arm is recorded and later arms still run')
    return ap.parse_args()


args = parse_args()
assert args.tiny_steps >= 0 and args.real_steps >= 0 and args.seq >= 2 and args.real_batch >= 0
RUNS = []
REPO_DIR = Path(__file__).resolve().parent
ARMS_SPEC = None
if args.arms:
    ARMS_SPEC = json.loads((REPO_DIR / args.arms).read_text(encoding='utf-8'))
    for arm in ARMS_SPEC['arms']:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', arm['name']), arm['name']
        assert all(k.startswith('K3_') for k in arm.get('env', {})), 'arm env may only set K3_* knobs'
        RUNS.append(dict(mode='arm', name=arm['name'], env={k: str(v) for k, v in arm.get('env', {}).items()},
                         timeout_sec=int(arm.get('timeout_sec', 0)), note=arm.get('note', '')))
elif args.script != 'experiment.py':
    # Alternative pipeline scripts own their stages; the driver just provides env/venv/zip.
    RUNS.append(dict(mode='staged', steps=0, batch=0))
else:
    if args.profile in ('all', 'tiny'):
        RUNS.append(dict(mode='tiny', steps=args.tiny_steps, batch=1))
    if args.profile in ('all', 'real'):
        RUNS.append(dict(mode='real', steps=args.real_steps, batch=args.real_batch))
assert RUNS, 'empty run plan'
sys.path.insert(0, str(REPO_DIR))
import mem_guard
GUARD = mem_guard.config((ARMS_SPEC or {}).get('memory_guard'))
TOTAL_BUDGET = int((ARMS_SPEC or {}).get('total_budget_sec', 0))   # whole-session wall clock for arm runs (0 = off)
DRIVER_T0 = time.time()
RUN_ID = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:6]
RUN = Path(args.run_root) / (args.profile + '-' + RUN_ID)
ART = RUN / 'artifacts'
ART.mkdir(parents=True)
DRIVE = None
if args.drive_base:
    DRIVE = Path(args.drive_base) / RUN.name
    DRIVE.mkdir(parents=True, exist_ok=True)

SECRETS = [str(v) for k, v in os.environ.items() if len(str(v)) >= 8 and re.search('TOKEN|SECRET|API_KEY|PASSWORD', k, re.I)]
TOKEN = os.environ.get('HF_TOKEN', '')
if TOKEN:
    SECRETS.append(TOKEN)


def redact(s):
    for secret in sorted(set(SECRETS), key=len, reverse=True):
        s = s.replace(secret, '[REDACTED]')
    s = re.sub(r'hf_[A-Za-z0-9]{12,}', '[REDACTED]', s)
    s = re.sub(r'gh[pousr]_[A-Za-z0-9]{20,}', '[REDACTED]', s)
    s = re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[REDACTED]', s)
    return s


ZIP = RUN.with_suffix('.zip')
lock = threading.Lock()


def snapshot():
    with lock:
        entries = {}
        temp = ZIP.with_suffix('.tmp')
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED) as z:
            for p in sorted(RUN.rglob('*')):
                if not p.is_file() or p.suffix not in {'.json', '.jsonl', '.log', '.npz', '.py'}:
                    continue
                rel = p.relative_to(RUN).as_posix()
                # Read each file once: checksum covers exactly the bytes put in the ZIP.
                if p.suffix == '.npz':
                    sha = hashlib.sha256(); size = 0
                    with p.open('rb') as src, z.open(rel, 'w', force_zip64=True) as dst:
                        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                            sha.update(chunk); size += len(chunk); dst.write(chunk)
                    entries[rel] = {'size': size, 'sha256': sha.hexdigest()}
                else:
                    data = redact(p.read_text(encoding='utf-8', errors='replace')).encode('utf-8')
                    entries[rel] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                    z.writestr(rel, data)
            z.writestr('manifest.json', json.dumps({'run': RUN.name, 'repo_sha': args.repo_sha,
                                                    'k3_revision': args.k3_revision, 'files': entries}, indent=2))
        os.replace(temp, ZIP)
        sha = hashlib.sha256()
        with ZIP.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                sha.update(chunk)
        digest = sha.hexdigest()
        ZIP.with_suffix('.zip.sha256').write_text(digest + '  ' + ZIP.name + '\n')
        if DRIVE:
            try:
                dst = DRIVE / ZIP.name; pending = dst.with_suffix('.tmp')
                shutil.copyfile(ZIP, pending); os.replace(pending, dst)
                shutil.copyfile(ZIP.with_suffix('.zip.sha256'), DRIVE / (ZIP.name + '.sha256'))
            except Exception as e:
                print('[Drive archive copy failed]', type(e).__name__, '; local ZIP is preserved')
        return digest


def status(state, **kw):
    p = RUN / 'status.json'; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state': state, 'profile': args.profile, 'plan': RUNS, 'repo_sha': args.repo_sha, **kw}, indent=2))
    os.replace(tmp, p)


# Provenance copies: the archive carries the exact source bytes that ran.
(RUN / 'experiment.py').write_text((REPO_DIR / 'experiment.py').read_text(encoding='utf-8'), encoding='utf-8')
(RUN / 'colab_run.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
SCRIPT_NAME = Path(args.script).name
(RUN / SCRIPT_NAME).write_text((REPO_DIR / args.script).read_text(encoding='utf-8'), encoding='utf-8')
if args.arms:
    (RUN / 'arms.json').write_text((REPO_DIR / args.arms).read_text(encoding='utf-8'), encoding='utf-8')
for module in ('staged_core.py', 'mem_guard.py'):
    if (REPO_DIR / module).is_file():
        (RUN / module).write_text((REPO_DIR / module).read_text(encoding='utf-8'), encoding='utf-8')
(RUN / 'run_plan.json').write_text(json.dumps({'profile': args.profile, 'phases': RUNS, 'seq': args.seq,
                                               'repo_sha': args.repo_sha, 'k3_revision': args.k3_revision,
                                               'source_sha256': {n: hashlib.sha256((REPO_DIR / n).read_bytes()).hexdigest()
                                                                 for n in ('experiment.py', 'colab_run.py')}}, indent=2), encoding='utf-8')
status('PREPARING'); snapshot()
venv = Path(args.venv_root) / f'k3_env_v2_py{sys.version_info.major}{sys.version_info.minor}'
python = Path(sys.executable) if args.no_venv else venv / 'bin' / 'python'
stop = threading.Event()
archive_errors = []


def periodic():
    while not stop.wait(60):
        try:
            snapshot()
        except Exception as e:
            archive_errors.append(type(e).__name__); print('[archive retry]', type(e).__name__)


thread = threading.Thread(target=periodic, daemon=True); thread.start()
LAST_COMMAND = None; LAST_RETURN_CODE = None
CURRENT_PROC = None


def stop_current_arm(reason):
    p = CURRENT_PROC
    if p is not None and p.poll() is None:
        print('[mem-guard] stopping current phase:', reason, flush=True)
        mem_guard.kill_tree(p)


WATCHED_DISKS = [d for d in (args.run_root, args.venv_root, os.environ.get('K3_CACHE_DIR', '')) if d]
MONITOR = mem_guard.Monitor(GUARD, RUN / 'memory_monitor.jsonl', WATCHED_DISKS, on_overflow=stop_current_arm).start()


def execute(cmd, env=None, timeout_sec=0):
    global LAST_COMMAND, LAST_RETURN_CODE, CURRENT_PROC
    LAST_COMMAND = [str(c) for c in cmd]; LAST_RETURN_CODE = None
    proc = None; killer = None; drained = threading.Event()
    try:
        with (RUN / 'run.log').open('a', encoding='utf-8') as log:
            log.write(redact('[command] ' + repr(LAST_COMMAND)) + '\n'); log.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    errors='replace', start_new_session=(os.name == 'posix'))
            CURRENT_PROC = proc

            def _reap(p=proc):   # a grandchild that outlives the phase would keep the pipe (and the TPU) open
                p.wait()
                if not drained.wait(30):
                    mem_guard.kill_tree(p, grace=5)
            threading.Thread(target=_reap, daemon=True).start()
            if timeout_sec:   # hard per-phase wall clock: a hung compile/arm must not eat the whole session
                def _kill(p=proc):
                    if p.poll() is None:
                        log.write(f'[timeout] phase exceeded {timeout_sec}s; terminating\n'); log.flush(); mem_guard.kill_tree(p)
                killer = threading.Timer(timeout_sec, _kill); killer.daemon = True; killer.start()
            for line in proc.stdout:
                safe = redact(line); log.write(safe); log.flush(); print(safe, end='', flush=True)
            drained.set(); LAST_RETURN_CODE = proc.wait()
            if killer is not None: killer.cancel()
            log.write(f'[exit_code] {LAST_RETURN_CODE}\n'); log.flush()
        return LAST_RETURN_CODE
    finally:
        drained.set(); CURRENT_PROC = None
        if proc is not None:
            mem_guard.kill_tree(proc, grace=5 if proc.poll() is not None else 30)   # also clears stray group members


def prepare_environment(venv, python):
    # Colab can lack ensurepip. Use parent pip to seed an initially pip-free venv.
    # Never delete the previous environment; v2 uses a new versioned path.
    status('CREATING_VENV'); snapshot()
    cmd = [sys.executable, '-m', 'venv', '--without-pip', str(venv)]
    rc = execute(cmd)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    status('BOOTSTRAPPING_PIP'); snapshot()
    cmd = [sys.executable, '-m', 'pip', '--python', str(python), 'install', '--upgrade', 'pip']
    rc = execute(cmd)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    rc = execute([str(python), '-c', 'import sys,pip; assert sys.prefix != sys.base_prefix; print("[venv] ready",sys.version,pip.__version__)'])
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


RESETS = []


def install_environment(venv, python):
    prepare_environment(venv, python)
    status('INSTALLING'); snapshot()
    rc = execute([str(python), '-m', 'pip', 'install', *PACKAGES])
    if rc:
        raise RuntimeError('Dependency installation failed; see archived run.log')


def full_reset(reasons, before_arm):
    """One reset before the next process: kill leftovers, free disk if needed, fresh venv + reinstall."""
    global venv, python
    event = dict(before_arm=before_arm, reasons=reasons, t=round(time.time()))
    print('[mem-guard] full reset before', before_arm, '|', '; '.join(reasons), flush=True)
    event['holders'] = mem_guard.clear_holders()
    cache = os.environ.get('K3_CACHE_DIR', '')
    if cache and any(r.startswith('disk_used') for r in reasons):
        shutil.rmtree(cache, ignore_errors=True); event['tensor_cache_removed'] = True
    if args.no_venv:
        event['reinstall'] = 'skipped (--no-venv)'
    else:
        old_venv, old_python = venv, python
        new_venv = old_venv.with_name(old_venv.name.split('_reset')[0] + f'_reset{len(RESETS) + 1}')
        try:
            install_environment(new_venv, new_venv / 'bin' / 'python')
            venv, python = new_venv, new_venv / 'bin' / 'python'
            shutil.rmtree(old_venv, ignore_errors=True)     # created by this run
            event['reinstall'] = 'ok'
        except Exception as e:
            venv, python = old_venv, old_python
            event['reinstall'] = f'failed ({type(e).__name__}); previous environment kept'
    reasons_after, snap = mem_guard.idle_check(GUARD, WATCHED_DISKS)
    event['still_over_after_reset'] = reasons_after; event['after'] = snap
    RESETS.append(event)
    (RUN / 'memory_resets.json').write_text(json.dumps(RESETS, indent=2), encoding='utf-8')
    snapshot()


def run_arm(phase, base_env, arm_results):
    """One arm as an isolated process, with the memory guard and the session time budget."""
    name = phase['name']; timeout = phase['timeout_sec']
    if arm_results:                                       # idle state left by the previous process
        time.sleep(GUARD['settle_sec'])
        reasons, snap = mem_guard.idle_check(GUARD, WATCHED_DISKS)
        print('[mem-guard] idle before', name, json.dumps(snap.get('host')), 'reasons:', reasons or 'none', flush=True)
        if reasons and len(RESETS) < GUARD['max_resets']:
            full_reset(reasons, name)
    for attempt in (1, 2):
        if TOTAL_BUDGET:
            remaining = int(TOTAL_BUDGET - (time.time() - DRIVER_T0) - 180)
            if remaining < 600:
                return dict(name=name, exit_code=None, seconds=0, state='SKIPPED_NO_TIME')
            timeout = min(phase['timeout_sec'] or remaining, remaining)
        env = base_env.copy(); env.update(phase['env'])
        env['K3_ARTIFACTS'] = str(ART / name)
        env['K3_HBM_DIRTY_FRAC'] = str(GUARD['hbm_dirty_frac'])
        if timeout:
            env['K3_ARM_DEADLINE_SEC'] = str(max(timeout - 600, int(timeout * 0.7)))
        status('RUNNING_ARM', arm=name, attempt=attempt, arms=arm_results, resets=len(RESETS)); snapshot()
        MONITOR.label = name; t0 = time.time()
        code = execute([str(python), str(RUN / SCRIPT_NAME)], env, timeout)
        MONITOR.label = 'idle'
        if code == mem_guard.HBM_DIRTY_EXIT and attempt == 1 and len(RESETS) < GUARD['max_resets']:
            full_reset(['accelerator memory occupied at arm start'], name)
            continue
        break
    return dict(name=name, exit_code=code, seconds=round(time.time() - t0), attempts=attempt,
                state='COMPLETED' if code == 0 else 'FAILED')


def record_failure(exc):
    details = redact(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    (RUN / 'error.log').write_text(details, encoding='utf-8')
    status('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED', error_type=type(exc).__name__,
           error=redact(str(exc)), last_command=[redact(c) for c in LAST_COMMAND] if LAST_COMMAND else None,
           exit_code=LAST_RETURN_CODE)
    print('[run stopped]', details)


code = None
try:
    if not args.no_venv:
        install_environment(venv, python)
    base_env = os.environ.copy()
    base_env.update(K3_SEQ=str(args.seq), K3_ARTIFACTS=str(ART), K3_REVISION=args.k3_revision, PYTHONUNBUFFERED='1')
    if TOKEN:
        base_env['HF_TOKEN'] = TOKEN
    if 'COLAB_TPU_ADDR' in base_env:
        base_env.setdefault('JAX_PLATFORMS', 'tpu,cpu')
    arm_results = []
    for phase in RUNS:
        if phase['mode'] == 'arm':
            arm_results.append(run_arm(phase, base_env, arm_results))
            r = arm_results[-1]
            print(f"[arm] {r['name']} -> {r['state']} (exit {r['exit_code']}, {r['seconds']}s)", flush=True)
            status('RUNNING_ARMS', arms=arm_results, resets=len(RESETS)); snapshot()
            continue                                   # a failed arm never stops later arms
        env = base_env.copy()
        env.update(K3_MODE=phase['mode'], K3_STEPS=str(phase['steps']), K3_BATCH=str(phase['batch']),
                   K3_ARTIFACTS=str(ART / phase['mode']))
        status('RUNNING_' + phase['mode'].upper()); snapshot()
        code = execute([str(python), str(RUN / SCRIPT_NAME)], env)
        if code:
            raise RuntimeError(f"phase {phase['mode']} failed with exit code {code}; later phases stopped")
    if arm_results:
        ok = all(a['state'] == 'COMPLETED' for a in arm_results)
        status('COMPLETED' if ok else 'COMPLETED_WITH_FAILED_ARMS', exit_code=0, arms=arm_results, resets=len(RESETS),
               host_ram_peak_pct=MONITOR.peak, archive_errors=archive_errors)
    else:
        status('COMPLETED', exit_code=0, archive_errors=archive_errors)
except BaseException as exc:
    record_failure(exc)
finally:
    MONITOR.stop(); stop.set(); thread.join(timeout=10)
    digest = snapshot()
    print('\nRESULT ZIP:', ZIP, '\nSHA256:', digest)
    if DRIVE:
        print('DRIVE:', DRIVE / ZIP.name)
