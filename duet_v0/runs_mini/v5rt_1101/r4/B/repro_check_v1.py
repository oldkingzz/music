"""Round 4 phase 1: live bench v2 reproduces v1 exactly for --lat wired and --lat bt (run v5rt_1101). Compares the full
digest and every JSON field except timing / time / code between each v1 file and its v2 re-run, and checks the v2
file's code hashes. Writes a new JSON (mode 'x'). Val only; no sound; reads files only.

  .venv/bin/python runs_mini/v5rt_1101/r4/B/repro_check_v1.py --out runs_mini/v5rt_1101/r4/B/repro_check_v1.json
"""
import argparse
import hashlib
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
LBD = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench')
V2 = os.path.join(ABS, 'futures', 'v5_rt', 'live_bench_rt_v2.py')
PAIRS = [('L_reactive_r2_mix_v2_sung_wired.json', 'L4_v2repro_reactive_r2_mix_v2_sung_wired.json'),
         ('L_reactive_r2_mix_v2_sung_bt.json', 'L4_v2repro_reactive_r2_mix_v2_sung_bt.json'),
         ('R3X_rt_r3_X_regtrkst_v1_sung_wired.json', 'L4_v2repro_rt_r3_X_regtrkst_v1_sung_wired.json'),
         ('R3X_rt_r3_X_regtrkst_v1_sung_bt.json', 'L4_v2repro_rt_r3_X_regtrkst_v1_sung_bt.json')]
SKIP = ('timing', 'time', 'code')


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    v2sha = sha(V2)
    rows, ok = [], True
    for f1, f2 in PAIRS:
        with open(os.path.join(LBD, f1)) as fh:
            r1 = json.load(fh)
        with open(os.path.join(LBD, f2)) as fh:
            r2 = json.load(fh)
        same_digest = r1['digest'] == r2['digest']
        diff = sorted(k for k in set(r1) | set(r2) if k not in SKIP and r1.get(k) != r2.get(k))
        code_ok = (r2['code'].get('live_bench_rt_v2.py') == v2sha
                   and all(r2['code'].get(k) == r1['code'].get(k) for k in r1['code']))
        row = {'v1': f1, 'v2': f2, 'digest_v1': r1['digest'], 'digest_v2': r2['digest'], 'same_digest': same_digest,
               'differing_fields': diff, 'code_ok': code_ok, 'code_v2': r2['code'],
               'rt_nd_live': [r1['headline']['rt_nd_live'], r2['headline']['rt_nd_live']]}
        ok = ok and same_digest and not diff and code_ok
        rows.append(row)
        print(f"{f2}: digest {'SAME' if same_digest else 'DIFFERENT'} ({r2['digest'][:16]}), other differing fields "
              f"{diff or 'none'}, code {'ok' if code_ok else 'MISMATCH'}")
    out = {'ok': ok, 'live_bench_rt_v2_sha256': v2sha, 'pairs': rows,
           'note': 'v2 = thin wrapper around frozen v1; val only, no test split'}
    with open(a.out, 'x') as fh:
        json.dump(out, fh, indent=1)
    print('ALL REPRODUCED' if ok else 'NOT REPRODUCED', flush=True)


if __name__ == '__main__':
    main()
