"""How much of the bench_rt -> live gap is the bar position (round 3 phase 1 diagnostic). The page gives the engine
tonic, M, K and bpm only, so RTDuet starts every system with pos = k % M; 14 of the 17 val chorales start with a pickup,
where that is not the song's bar position. bench_rt.simulate / score_song (bench_rt's own functions, bench3's heard
melody) on the 17 val chorales for each engine with the true pos and with pos = k % M, at L = 0.1 and at the live lag
L = 0.081 beat (44 ms + Lo 10 ms at 90 bpm). Val only, tuned on val, no test split. Writes one new JSON.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_pos_effect_v1.py --out runs_mini/v5rt_1101/live_bench/<new>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import bench_rt as BR  # noqa: E402

LB = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/live_bench'
SPECS = ('reactive:r2_mix_v2', 'rt:r2_copyhmbold_v1', 'rt:r2_copy_v1', 'rt:r1_C_revoice_v2')


def run(obj, L, pos_mode):
    acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                         'on_time_right', 'changes', 'late_entry'), 0.0)
    for s in BR.bench3.eval_sets('val')['bach']:
        notes = [(float(on), float(off), int(m)) for on, off, m in sorted(BR.HEARD1(s)['sop'], key=lambda n: n[0])]
        pos = [b['pos'] for b in s['beats']] if pos_mode == 'true' else [k % s['M'] for k in range(s['K'])]
        T, H, _, _ = BR.simulate(obj, {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': pos}, notes, L)
        BR.score_song(s, T, H, L, acc)
    return acc['hit'] / acc['d']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(LB + os.sep) or os.path.exists(path):
        sys.exit('--out must be a new file under ' + LB)
    train = BR.bench3.train_sets()
    res = {'note': 'val bach only (17 chorales), bench3 heard melody, tuned on val, no test split', 'rows': {}}
    for spec in SPECS:
        obj = BR.load(spec, train)
        row = {}
        for L in (0.1, 0.081):
            for pm in ('true', 'k%M'):
                row[f'L{L} pos {pm}'] = run(obj, L, pm)
        res['rows'][spec] = row
        print(spec, json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
