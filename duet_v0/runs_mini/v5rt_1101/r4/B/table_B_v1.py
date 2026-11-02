"""Round 4 phase 1 table (run v5rt_1101): live bench v2 baselines, Q1 (calibration effect), C2 for W, K1/K2 reference
values, bench_rt / attack_metric at L = 0.5 and 0.3 (K4 reference), W's calibration-error robustness. Reads result
files only; writes one new JSON (mode 'x'). Val only (17 val Bach chorales; bench_rt: val bach/folk/pop/pop909),
systems tuned on val, no test split.

bt on the sung voice reuses the round-3 v1 files (v2 reproduces them bit-identically: r4/B/repro_check_v1.json); wired
is the round-3 v1 files (reference only).

  .venv/bin/python runs_mini/v5rt_1101/r4/B/table_B_v1.py --out runs_mini/v5rt_1101/r4/B/table_B_v1.json
"""
import argparse
import json
import os

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
LBD = os.path.join(RUN, 'live_bench')
ENG = {'reactive:r2_mix_v2': 'reactive_r2_mix_v2', 'rt:r3_X_regtrkst_v1': 'rt_r3_X_regtrkst_v1',
       'rt:r3_X_regtrkdk_v1': 'rt_r3_X_regtrkdk_v1', 'rt:r3_X_regtrkstsafe_v1': 'rt_r3_X_regtrkstsafe_v1'}
SHORT = {'reactive:r2_mix_v2': 'reactive', 'rt:r3_X_regtrkst_v1': 'W', 'rt:r3_X_regtrkdk_v1': 'regtrkdk',
         'rt:r3_X_regtrkstsafe_v1': 'regtrkstsafe'}
VOICES = ('sung', 'sung_low')
LATS = ('wired', 'micbt', 'micbt_cal', 'bt', 'bt_cal')
ROB = ('micbt_cal_m25', 'micbt_cal_p25', 'bt_cal_m25', 'bt_cal_p25')
COLS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')
W = 'rt:r3_X_regtrkst_v1'
R = 'reactive:r2_mix_v2'


def live_file(spec, voice, lat):
    n = ENG[spec]
    if lat == 'wired' or (lat == 'bt' and voice == 'sung'):
        name = (f'L_{n}_{voice}_{lat}.json' if spec == R else f'R3X_{n}_{voice}_{lat}.json')
        return name, 'v1 file (round 3)'
    return f'L4_{n}_{voice}_{lat}.json', 'v2 run (round 4)'


def load(path):
    with open(path) as fh:
        return json.load(fh)


def cell(spec, voice, lat):
    name, src = live_file(spec, voice, lat)
    r = load(os.path.join(LBD, name))
    assert (r['spec'], r['voice'], r['lat'], r['split'], r['songs_n']) == (spec, voice, lat, 'val', 17), name
    out = {k: r['headline'][k] for k in COLS}
    out.update({'fresh': r['headline']['fresh'], 'notes': r['headline']['notes'], 'file': 'live_bench/' + name,
                'source': src, 'digest': r['digest'], 'latency': r['latency']})
    return out


def mean(a):
    return sum(a) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cells = {s: {v: {l: cell(s, v, l) for l in LATS} for v in VOICES} for s in ENG}
    for v in VOICES:
        for l in ROB:
            cells[W][v][l] = cell(W, v, l)
    q1 = {}
    for s in (R, W):
        q1[SHORT[s]] = {v: {f'{cal} - {raw}': {k: cells[s][v][cal][k] - cells[s][v][raw][k] for k in COLS}
                            for raw, cal in (('bt', 'bt_cal'), ('micbt', 'micbt_cal'))} for v in VOICES}
    m = lambda s, l: mean([cells[s][v][l]['rt_nd_live'] for v in VOICES])  # noqa: E731
    c2 = {'W_mean_bt': m(W, 'bt'), 'W_mean_bt_cal': m(W, 'bt_cal'), 'W_mean_micbt': m(W, 'micbt'),
          'W_mean_micbt_cal': m(W, 'micbt_cal')}
    c2['holds'] = c2['W_mean_bt_cal'] >= c2['W_mean_bt'] and c2['W_mean_micbt_cal'] >= c2['W_mean_micbt']
    k1 = {}
    for s in ENG:
        k1[SHORT[s]] = {'K1_mean': mean([cells[s][v][l]['rt_nd_live'] for v in VOICES for l in ('micbt_cal', 'bt_cal')]),
                        'fresh_plus_stumbles_4cells': sum(cells[s][v][l]['fresh_per100'] + cells[s][v][l]['stumbles_per100']
                                                          for v in VOICES for l in ('micbt_cal', 'bt_cal')),
                        'wired_mean': m(s, 'wired')}
    wk1 = k1['W']['K1_mean']
    for s in ('rt:r3_X_regtrkdk_v1', 'rt:r3_X_regtrkstsafe_v1'):
        k2 = all(cells[s][v][l]['harsh'] <= cells[W][v][l]['harsh'] + 0.003
                 and cells[s][v][l]['fresh_per100'] <= cells[W][v][l]['fresh_per100'] + 0.5
                 and cells[s][v][l]['stumbles_per100'] <= cells[W][v][l]['stumbles_per100'] + 1.0
                 for v in VOICES for l in ('micbt_cal', 'bt_cal'))
        k1[SHORT[s]].update({'K1_vs_W': k1[SHORT[s]]['K1_mean'] - wk1,
                             'K1_pass': k1[SHORT[s]]['K1_mean'] >= wk1 + 0.005, 'K2_pass_vs_W': k2})
    brt = {}
    for L in ('0.5', '0.3', '0.1'):
        for s in (R, W):
            n = ENG[s]
            if L == '0.1':
                f1 = 'B_reactive_r2_mix_v2_L0.1.json' if s == R else 'R3X_rt_r3_X_regtrkst_v1_L0.1.json'
                f2 = 'B_reactive_r2_mix_v2_L0.1_seed2.json' if s == R else 'R3X_rt_r3_X_regtrkst_v1_L0.1_seed2.json'
                fa = 'reactive_r2_mix_v2_L0.1.json' if s == R else 'R3X_rt_r3_X_regtrkst_v1_L0.1.json'
            else:
                f1, f2, fa = f'L4_{n}_L{L}.json', f'L4_{n}_L{L}_seed2.json', f'L4_{n}_L{L}.json'
            s1, s2 = load(os.path.join(RUN, 'scores', f1)), load(os.path.join(RUN, 'scores', f2))
            at = load(os.path.join(RUN, 'attack', fa))
            assert float(s1['L']) == float(L) and float(s2['L']) == float(L) and float(at['L']) == float(L)
            brt.setdefault('L' + L, {})[SHORT[s]] = {
                'rt_nd': s1['headline']['rt_nd'], 'harsh': s1['headline']['harsh'],
                'rt_nd_seed2': s2['headline']['rt_nd'], 'fresh_clash_per100': at['headline']['fresh_clash_per100'],
                'bach': s1['rows']['bach']['rt_nd'], 'files': ['scores/' + f1, 'scores/' + f2, 'attack/' + fa]}
    for L, d in brt.items():
        w, r = d['W'], d['reactive']
        d['W_minus_reactive'] = {'rt_nd': w['rt_nd'] - r['rt_nd'], 'harsh': w['harsh'] - r['harsh'],
                                 'seed2': w['rt_nd_seed2'] - r['rt_nd_seed2'],
                                 'fresh': w['fresh_clash_per100'] - r['fresh_clash_per100']}
        d['W_passes_K4_form'] = (w['rt_nd'] >= r['rt_nd'] + 0.005 and w['harsh'] <= r['harsh']
                                 and w['fresh_clash_per100'] <= r['fresh_clash_per100']
                                 and w['rt_nd_seed2'] - r['rt_nd_seed2'] >= 0.003)
    rob = {l: mean([cells[W][v][l]['rt_nd_live'] for v in VOICES]) for l in ('micbt_cal',) + ROB[:2] + ('bt_cal',) + ROB[2:]}
    out = {'note': 'val only (17 val Bach chorales live; bench_rt val 4 datasets), tuned on val, no test split',
           'cells': {SHORT[s]: c for s, c in cells.items()}, 'q1': q1, 'c2': c2, 'k1_reference': k1,
           'bench_rt': brt, 'robustness_W_mean_sung_sunglow': rob}
    with open(a.out, 'x') as fh:
        json.dump(out, fh, indent=1)
    print('engine        voice     lat           rt_nd_live harsh  fresh/100 stumbles/100  file')
    for s in ENG:
        for v in VOICES:
            for l, c in cells[s][v].items():
                print(f"{SHORT[s]:13s} {v:9s} {l:13s} {c['rt_nd_live']:.4f}     {c['harsh']:.4f} {c['fresh_per100']:5.2f}    "
                      f"{c['stumbles_per100']:5.2f}        {c['file']} ({c['source']})")
    print('\nQ1 (calibrated minus uncalibrated):')
    for e, d in q1.items():
        for v, dd in d.items():
            for k, x in dd.items():
                print(f"  {e:9s} {v:9s} {k:20s} rt_nd {x['rt_nd_live']:+.4f} harsh {x['harsh']:+.4f} fresh "
                      f"{x['fresh_per100']:+.2f} stumbles {x['stumbles_per100']:+.2f}")
    print('C2:', json.dumps(c2))
    print('K1 reference:', json.dumps(k1))
    for L, d in brt.items():
        print(f"bench_rt {L}: reactive rt_nd {d['reactive']['rt_nd']:.4f} harsh {d['reactive']['harsh']:.4f} seed2 "
              f"{d['reactive']['rt_nd_seed2']:.4f} fresh {d['reactive']['fresh_clash_per100']:.2f} | W rt_nd "
              f"{d['W']['rt_nd']:.4f} harsh {d['W']['harsh']:.4f} seed2 {d['W']['rt_nd_seed2']:.4f} fresh "
              f"{d['W']['fresh_clash_per100']:.2f} | W - reactive {json.dumps({k: round(x, 4) for k, x in d['W_minus_reactive'].items()})}"
              f" | W passes the K4 form: {d['W_passes_K4_form']}")
    print('robustness W mean(sung, sung_low):', json.dumps({k: round(x, 4) for k, x in rob.items()}))


if __name__ == '__main__':
    main()
