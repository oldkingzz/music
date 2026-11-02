"""Round 3 phase 1 sanity checks (b) and (c) for the live bench (val only; tuned on val, no test split). Read-only on
the live_bench JSONs; writes one new JSON (mode 'x').

(b) phone_twinkle: live_rt_eval_v2 samples Duet's note every 10 ms, takes a block's note as heard 20 ms + Lo after the
    block started (10 ms later than the voice writes it) and stops each of her notes 30 ms before its offset. The exact
    metric of live_bench_rt reads Duet's notes from the voice and counts her full notes. Here the exact metric is
    recomputed with each of those two conventions (Duet + 10 ms; her notes - 30 ms) and both, from the notes saved in
    the L_*_phone_twinkle.json files, to show where the difference comes from.
(c) clean voice vs bench_rt: reactive:r2_mix_v2 through bench_rt.simulate / score_song on the 17 val chorales
    (bench_rt's own functions; scored with bench_rt's exact rt_nd) under: bench_rt's setting (L = 0.1, bench3's 10 %
    synthetic mishearing, true pos) and then one change at a time: no mishearing; pos = k % M (what the live engine
    gives: no pickup setting on the page); L = the live lag (median detect lag + Lo, in beats at 90 bpm); the live
    tracker's own heard pitches (from L_reactive_r2_mix_v2_clean_wired.json) in place of the synthetic mishearing;
    and all live-like changes together.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_sanity_v1.py --out runs_mini/v5rt_1101/live_bench/<new>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
from songs import twinkle  # noqa: E402

BR = LBR.BR
LB = os.path.join(LBR.ABS, 'runs_mini', 'v5rt_1101', 'live_bench')


def sanity_b():
    out = {}
    song = twinkle()
    z = [b['z'] for b in song['beats']]
    t0h, T_b = 4.0, 0.75
    base = [(t0h + on * T_b, t0h + off * T_b, int(m), float(on), float(off))
            for on, off, m in sorted(song['sop'], key=lambda n: n[0])]
    for name in ('reactive_r2_mix_v2', 'rt_r2_copyhmbold_v1'):
        r = json.load(open(os.path.join(LB, f'L_{name}_phone_twinkle.json')))
        duet = [tuple(d) for d in r['phone']['duet_s']]
        rows = {}
        for label, dshift, cut in (('exact (live_bench_rt)', 0.0, 0.0), ('exact, Duet +10 ms', 0.01, 0.0),
                                   ('exact, her notes -30 ms', 0.0, 0.03), ('exact, both', 0.01, 0.03)):
            her = [(a, e - cut, m, on, off) for a, e, m, on, off in base]
            du = [(a + dshift, e + dshift, m) for a, e, m in duet]
            acc = LBR.new_acc()
            LBR.score(her, du, z, song['tonic'], t0h, T_b, acc)
            met = LBR.metrics(acc)
            rows[label] = {'rt_nd_live': met['rt_nd_live'], 'harsh': met['harsh']}
        rows['sampled (live_rt_eval_v2.truth)'] = {'rt_nd_live': r['phone']['sampled']['fit'],
                                                   'harsh': r['phone']['sampled']['harsh']}
        out[name] = rows
        print(name, json.dumps(rows), flush=True)
    return out


def bench_rt_bach(obj, L, notes_fn, pos_mode):
    acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                         'on_time_right', 'changes', 'late_entry'), 0.0)
    for s in LBR.bench3.eval_sets('val')['bach']:
        notes = notes_fn(s)
        pos = [b['pos'] for b in s['beats']] if pos_mode == 'true' else [k % s['M'] for k in range(s['K'])]
        meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': pos}
        T, H, _, _ = BR.simulate(obj, meta, notes, L)
        BR.score_song(s, T, H, L, acc)
    return acc['hit'] / acc['d'], acc['harsh'] / acc['d']


def sanity_c():
    live = json.load(open(os.path.join(LB, 'L_reactive_r2_mix_v2_clean_wired.json')))
    lag_ms = live['tracker']['bach']['detect_ms']['median']
    L_live = (lag_ms / 1000.0 + LBR.LAT['wired']['Lo']) / (60.0 / LBR.BPM)
    heard_by_id = {}
    for s in live['songs']:
        heard_by_id[s['id']] = [row[5] for row in s['tracker_rows']]

    def synth(s):
        return [(float(on), float(off), int(m)) for on, off, m in sorted(BR.HEARD1(s)['sop'], key=lambda n: n[0])]

    def true(s):
        return [(float(on), float(off), int(m)) for on, off, m in sorted(s['sop'], key=lambda n: n[0])]

    def tracker(s):
        hm = heard_by_id[s['id']]
        return [(float(on), float(off), int(m if h is None else h))
                for (on, off, m), h in zip(sorted(s['sop'], key=lambda n: n[0]), hm)]

    obj = BR.load('reactive:r2_mix_v2', LBR.bench3.train_sets())
    conds = [('bench_rt: L 0.1, synthetic 10 % mishearing, true pos', 0.1, synth, 'true'),
             ('no mishearing (true pitches)', 0.1, true, 'true'),
             ('pos = k % M (no pickup on the page)', 0.1, synth, 'k%M'),
             (f'L = live lag {L_live:.3f} beat ({lag_ms:.0f} ms + Lo 10 ms at 90 bpm)', L_live, synth, 'true'),
             ('live tracker pitches (clean) instead of synthetic', 0.1, tracker, 'true'),
             (f'all live-like: L {L_live:.3f}, tracker pitches, pos = k % M', L_live, tracker, 'k%M')]
    rows = {}
    for label, L, fn, pm in conds:
        nd, harsh = bench_rt_bach(obj, L, fn, pm)
        rows[label] = {'rt_nd': nd, 'harsh': harsh, 'L': L}
        print(f'{label:70s} rt_nd {nd:.4f} harsh {harsh:.4f}', flush=True)
    rows['live bench clean (RTDuet, exact)'] = {'rt_nd': live['headline']['rt_nd_live'], 'harsh': live['headline']['harsh']}
    print(f"{'live bench clean (RTDuet, exact)':70s} rt_nd {live['headline']['rt_nd_live']:.4f} "
          f"harsh {live['headline']['harsh']:.4f}", flush=True)
    return {'L_live': L_live, 'detect_ms_median': lag_ms, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(LB + os.sep) or os.path.exists(path):
        sys.exit('--out must be a new file under ' + LB)
    res = {'note': 'val only; systems tuned on val; no test split', 'b_phone_twinkle': sanity_b(),
           'c_clean_vs_bench_rt': sanity_c()}
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
