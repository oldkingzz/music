"""Round-4 select step (run v5rt_1101): is a round-4 variant decision-for-decision W outside its lag regime?
table_X_v1.py compared the live files' 'digest' fields, but the live bench's digest hashes the spec string too, so two
systems never share a digest even when every decision is the same (table_v1.json's 'same_as_W' digest flags are
therefore always False for a non-W system and mean nothing). This script compares the files themselves: every field
except spec, digest, timing, time, code and note (live bench), and every field except spec and the us_* timings
(bench_rt score files and attack files), recursively; it lists the first differing paths. Reads only files on disk.
Val only, tuned on val (this bundle has no test split).

  same_as_W_v1.py --out <new json under runs_mini/v5rt_1101/r4/X/>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
W = 'r3_X_regtrkst_v1'
VARS = {'R4G_': ('r4_G_ehedgesafe_v1', 'r4_G_ehedge_v1', 'r4_G_hedge_v1'),
        'R4H_': ('r4_H_lagk15hd_v1', 'r4_H_lagk15_v1', 'r4_H_lagk15hq_v1', 'r4_H_lagk2hd_v1'),
        'R4X_': ('r4_X_hgsafe_v1', 'r4_X_hdgsafe_v1', 'r4_X_hgeh_v1')}
LIVE_SKIP = {'spec', 'digest', 'timing', 'time', 'code', 'note'}


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def diff(a, b, path, out, skip_us):
    if len(out) >= 8:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if skip_us and str(k).startswith('us_'):
                continue
            if k not in a or k not in b:
                out.append(f'{path}/{k} (missing on one side)')
            else:
                diff(a[k], b[k], f'{path}/{k}', out, skip_us)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f'{path} (length {len(a)} vs {len(b)})')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f'{path}[{i}]', out, skip_us)
    elif a != b:
        out.append(f'{path}: {a!r} vs {b!r}'[:160])


def compare(rel_a, rel_b, live):
    a, b = load(rel_a), load(rel_b)
    if live:
        a = {k: v for k, v in a.items() if k not in LIVE_SKIP}
        b = {k: v for k, v in b.items() if k not in LIVE_SKIP}
    else:
        a = {k: v for k, v in a.items() if k != 'spec'}
        b = {k: v for k, v in b.items() if k != 'spec'}
    out = []
    diff(a, b, '', out, not live)
    return {'same': not out, 'first_diffs': out, 'files': [rel_a, rel_b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'r4', 'X') + os.sep) or os.path.exists(out_path):
        sys.exit('--out must be a new file under runs_mini/v5rt_1101/r4/X/')
    res = {'note': 'val only, tuned on val (no test split); W = r3_X_regtrkst_v1; W\'s wired / v11 / L0.1 files are '
                   'round 3\'s (R3X_), its micbt_cal / bt_cal / L0.5 files phase 1\'s (L4_)', 'systems': {}}
    for p, names in VARS.items():
        for n in names:
            r = {}
            for v in ('sung', 'sung_low'):
                r[f'{v}@wired'] = compare(f'live_bench/{p}rt_{n}_{v}_wired.json',
                                          f'live_bench/R3X_rt_{W}_{v}_wired.json', True)
                for lat in ('micbt_cal', 'bt_cal'):
                    r[f'{v}@{lat}'] = compare(f'live_bench/{p}rt_{n}_{v}_{lat}.json',
                                              f'live_bench/L4_rt_{W}_{v}_{lat}.json', True)
            r['phone_v11'] = compare(f'live_bench/{p}rt_{n}_phone_v11.json', f'live_bench/R3X_rt_{W}_phone_v11.json',
                                     True)
            for L, wp in (('0.1', 'R3X_'), ('0.5', 'L4_')):
                r[f'bench_rt L{L}'] = compare(f'scores/{p}rt_{n}_L{L}.json', f'scores/{wp}rt_{W}_L{L}.json', False)
                r[f'bench_rt L{L} seed2'] = compare(f'scores/{p}rt_{n}_L{L}_seed2.json',
                                                    f'scores/{wp}rt_{W}_L{L}_seed2.json', False)
                r[f'attack L{L}'] = compare(f'attack/{p}rt_{n}_L{L}.json', f'attack/{wp}rt_{W}_L{L}.json', False)
            res['systems'][n] = r
            same = [k for k, x in r.items() if x['same']]
            diff_k = [k for k, x in r.items() if not x['same']]
            print(f'{n:22s} same as W: {same}\n{"":22s} differs:   {diff_k}')
            for k in diff_k:
                if k.endswith(('wired', 'micbt_cal', 'phone_v11', 'L0.1', 'L0.1 seed2')):
                    print(f'{"":24s}{k}: {r[k]["first_diffs"][:3]}')
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
