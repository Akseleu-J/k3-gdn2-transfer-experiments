"""Unit checks for mem_guard (stdlib only; fake /proc readers, real child processes)."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mem_guard as mg

results = []


def check(name, ok, detail=''):
    assert ok, (name, detail)
    results.append(name); print(f'[PASS] {name} {detail}')


tmp = Path(tempfile.mkdtemp())
(tmp / 'meminfo').write_text('MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 250 kB\nShmem: 10 kB\n')
m = mg.host_memory(tmp / 'meminfo')
check('host_memory_uses_available', m['used_pct'] == 75.0 and m['total'] == 1024000, str(m))
check('host_memory_missing_is_none', mg.host_memory(tmp / 'nope') is None)

cfg = mg.config({'host_used_reset_pct': 50, 'kill_samples': 2, 'host_used_kill_pct': 90})
try:
    mg.config({'typo': 1}); bad = False
except ValueError:
    bad = True
check('config_rejects_unknown_keys', bad)

mem = lambda pct: (lambda: dict(used_pct=pct, total=1, available=1, shmem=0))
clean, _ = mg.idle_check(cfg, [], holders_fn=lambda: {}, memory_fn=mem(10.0), disk_fn=lambda d: {'/x': 10.0})
check('idle_clean_no_reset', clean == [])
r, _ = mg.idle_check(cfg, [], holders_fn=lambda: {}, memory_fn=mem(55.0), disk_fn=lambda d: {})
check('idle_ram_over_limit_resets', len(r) == 1 and 'host_ram_used' in r[0], str(r))
r, _ = mg.idle_check(cfg, [], holders_fn=lambda: {}, memory_fn=mem(10.0), disk_fn=lambda d: {'/k': 91.0})
check('idle_disk_over_limit_resets', len(r) == 1 and 'disk_used' in r[0])
r, snap = mg.idle_check(cfg, [], holders_fn=lambda: {4242: {'target': '/dev/vfio/0', 'cmd': 'x'}}, memory_fn=mem(10.0), disk_fn=lambda d: {})
check('idle_leftover_holder_resets', len(r) == 1 and '4242' in r[0] and '4242' in snap['holders'])

events = []
seq = iter([50.0, 95.0, 50.0, 95.0, 95.0])
mon = mg.Monitor(cfg, tmp / 'mon.jsonl', [], on_overflow=events.append,
                 memory_fn=lambda: dict(used_pct=next(seq), total=1, available=1, shmem=0), disk_fn=lambda d: {})
for i in range(4):
    mon.sample()
check('monitor_needs_consecutive_samples', events == [], str(events))
mon.sample()
check('monitor_fires_after_consecutive_overflow', len(events) == 1, str(events))
rows = [json.loads(x) for x in (tmp / 'mon.jsonl').read_text().splitlines()]
check('monitor_logs_every_sample', len(rows) == 5 and mon.peak == 95.0)


class Dev:
    def __init__(self, s): self.s = s
    def memory_stats(self): return self.s


st = mg.hbm_stats([Dev({'bytes_in_use': 5, 'bytes_limit': 100, 'peak_bytes_in_use': 9, 'other': 1}),
                   Dev({'bytes_in_use': 50, 'bytes_limit': 100}), Dev(None)])
check('hbm_stats_shape', st[0] == {'bytes_in_use': 5, 'peak_bytes_in_use': 9, 'bytes_limit': 100} and st[2] is None)
check('hbm_dirty_flags_only_occupied', mg.hbm_dirty(st, 0.10) == [1] and mg.hbm_dirty([None], 0.1) == [])

p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
t0 = time.time(); mg.kill_tree(p, grace=10)
check('kill_tree_stops_child', p.poll() is not None and time.time() - t0 < 30)

(tmp / 'lock').write_text('x')
res = mg.clear_holders(holders_fn=lambda: {}, lockfile=tmp / 'lock')
check('lock_removed_only_when_unheld', res['lockfile_removed'] and not (tmp / 'lock').exists())
(tmp / 'lock').write_text('x')
res = mg.clear_holders(holders_fn=lambda: {999999999: {}}, lockfile=tmp / 'lock', log=lambda *a: None)
check('lock_kept_while_held', not res['lockfile_removed'] and (tmp / 'lock').exists())

print(f'ALL {len(results)} MEM-GUARD CHECKS PASS')
