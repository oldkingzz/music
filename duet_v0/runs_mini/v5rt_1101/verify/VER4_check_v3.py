"""VER4_check_v3 (verifier, run v5rt_1101, round 4). Reads JSON files on disk only (no system is run). Val only (the
systems are tuned on val; this bundle has no test split). Writes ONE new JSON (mode 'x') under runs_mini/v5rt_1101/verify/.
v3 = VER4_check_v1.py (stopped on a missing file before writing anything: the select agent never ran the pick at
micbt_cal_m25 / micbt_cal_p25, only at bt_cal_m25 / bt_cal_p25) with those two settings compared against W instead.
VER4_check_v2.py is an unused byte copy of v1 (made with cp -n, never run, never edited).

1. Reproduction: each of the verifier's own re-runs (prefix VER4_) against the file the round-4 agents reported:
   live bench (every field except timing / time / code; digest included), bench_rt (headline and rows except the us_*
   timing fields), attack_metric (every field except the bench path). Float equality, no tolerance.
2. Pick = W where the lag regimes are off: the verifier's own pick and W runs at micbt_cal and wired (both voices) and
   v11 compared field by field (all but spec / digest / timing / time / code: the per-song Duet notes included); the
   pick's micbt_cal_m25 / _p25 (sung) against W's phase-1 files; bench_rt L 0.1 seed 1 / seed 2 and attack L 0.1
   against round 3's verifier files for W.
3. The round-4 rule (state_R3.json round4_rule_fixed_now) recomputed from the verifier's numbers only (rule logic copied
   from runs_mini/v5rt_1101/r4/X/table_X_v1.py; G1 = keep_table_v1.row / checks, imported unchanged): K1-K4, K5 from
   VER4_causal_probe_hgeh_v1.json, the rank among the verifier's three re-scored merges plus every other band member
   read from r4/X/table_v1.json, and the 'use' decision (G1 / G2 / G3 at wired and the wired mean).
4. Song-level spread of the K1 gain: per chorale, pick minus W rt_nd_live in each of the 4 cells; paired bootstrap over
   the 17 chorales (10000 draws, seed 20261102) of the K1 delta (pooled by time per cell, as the headline).
5. Robustness: the pick, W (phase-1 files) and reactive (verifier runs) on sung at the four +-25 ms calibration-error
   settings.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER4_check_v3.py --out runs_mini/v5rt_1101/verify/VER4_check_v3.json
"""
import argparse
import json
import os
import random
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402  (row / checks, unchanged)

PICK, W, RS = 'r4_X_hgeh_v1', 'r3_X_regtrkst_v1', 'reactive_r2_mix_v2'
CELLS = (('sung', 'micbt_cal'), ('sung_low', 'micbt_cal'), ('sung', 'bt_cal'), ('sung_low', 'bt_cal'))
KEYS = ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')
ROB = ('micbt_cal_m25', 'micbt_cal_p25', 'bt_cal_m25', 'bt_cal_p25')
SKIP_LIVE = ('timing', 'time', 'code')


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def diff(a, b, path='', out=None, skip=()):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f'{path}/{k}: missing in {"a" if k not in a else "b"}')
            else:
                diff(a[k], b[k], f'{path}/{k}', out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f'{path}: len {len(a)} != {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f'{path}[{i}]', out, skip)
    elif a != b:
        out.append(f'{path}: {a!r} != {b!r}')
    return out


def brt_view(d):
    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith('us_')}
    return {'spec': d['spec'], 'L': d['L'], 'seed': d['seed'], 'split': d['split'], 'headline': strip(d['headline']),
            'rows': {ds: strip(r) for ds, r in d['rows'].items()}, 'bench_sha256': d.get('bench_sha256')}


def same_live(mine, theirs, skip_spec=False):
    a, b = load(mine), load(theirs)
    sk = SKIP_LIVE + (('spec', 'digest') if skip_spec else ())
    d = diff(a, b, skip=sk)
    return {'mine': mine, 'theirs': theirs, 'identical': not d, 'n_diff': len(d), 'first_diffs': d[:5],
            'digest_mine': a['digest'][:16], 'digest_theirs': b['digest'][:16]}


def same_brt(mine, theirs, skip_spec=False):
    a, b = brt_view(load(mine)), brt_view(load(theirs))
    d = diff(a, b, skip=('spec',) if skip_spec else ())
    return {'mine': mine, 'theirs': theirs, 'identical': not d, 'n_diff': len(d), 'first_diffs': d[:5]}


def same_atk(mine, theirs, skip_spec=False):
    a, b = load(mine), load(theirs)
    d = diff(a, b, skip=('bench',) + (('spec',) if skip_spec else ()))
    return {'mine': mine, 'theirs': theirs, 'identical': not d, 'n_diff': len(d), 'first_diffs': d[:5]}


def head(rel):
    d = load(rel)
    if d['voice'] == 'phone_v11':
        p = d['phone']['proxies']
        return {'fresh_heard': p['fresh_heard'], 'harsh_heard': p['harsh_heard']}
    assert d['songs_n'] == 17 and d['bpm'] == 90.0 and d['count_in'] == 4 and d['split'] == 'val', rel
    return {k: d['headline'][k] for k in KEYS}


def songs(rel):
    d = load(rel)
    return [(s['id'], s['metrics']['rt_nd_live'] * s['metrics']['counted_beats'], s['metrics']['counted_beats'],
             s['metrics']['fresh'], s['metrics']['stumbles_per100'] * s['metrics']['notes'] / 100.0,
             s['metrics']['notes']) for s in d['songs']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = os.path.abspath(ap.parse_args().out)
    if not out_path.startswith(os.path.join(RUN, 'verify') + os.sep) or not os.path.basename(out_path).startswith(
            'VER4_') or os.path.exists(out_path):
        sys.exit('--out must be a new VER4_*.json under runs_mini/v5rt_1101/verify/')
    res = {'note': 'val only, systems tuned on val, no test split; verifier re-runs = prefix VER4_'}

    # 1. reproduction ------------------------------------------------------------------------------------------
    rep = []
    for v in ('sung', 'sung_low'):
        for lat in ('micbt_cal', 'bt_cal', 'wired'):
            rep.append(same_live(f'live_bench/VER4_rt_{PICK}_{v}_{lat}.json', f'live_bench/R4X_rt_{PICK}_{v}_{lat}.json'))
        for lat in ('micbt_cal', 'bt_cal'):
            rep.append(same_live(f'live_bench/VER4_rt_{W}_{v}_{lat}.json', f'live_bench/L4_rt_{W}_{v}_{lat}.json'))
            for n in ('r4_X_hdgsafe_v1', 'r4_X_hgsafe_v1'):
                rep.append(same_live(f'live_bench/VER4_rt_{n}_{v}_{lat}.json', f'live_bench/R4X_rt_{n}_{v}_{lat}.json'))
        rep.append(same_live(f'live_bench/VER4_rt_{W}_{v}_wired.json', f'live_bench/R3X_rt_{W}_{v}_wired.json'))
        rep.append(same_live(f'live_bench/VER4_{RS}_{v}_wired.json', f'live_bench/L_{RS}_{v}_wired.json'))
    for lat in ('bt_cal_m25', 'bt_cal_p25'):
        rep.append(same_live(f'live_bench/VER4_rt_{PICK}_sung_{lat}.json', f'live_bench/R4X_rt_{PICK}_sung_{lat}.json'))
    rep.append(same_live(f'live_bench/VER4_rt_{PICK}_phone_v11.json', f'live_bench/R4X_rt_{PICK}_phone_v11.json'))
    rep.append(same_live(f'live_bench/VER4_rt_{W}_phone_v11.json', f'live_bench/R3X_rt_{W}_phone_v11.json'))
    rep.append(same_live(f'live_bench/VER4_{RS}_phone_v11.json', f'live_bench/L_{RS}_phone_v11.json'))
    for sfx in ('', '_seed2'):
        rep.append(same_brt(f'scores/VER4_rt_{PICK}_L0.5{sfx}.json', f'scores/R4X_rt_{PICK}_L0.5{sfx}.json'))
        rep.append(same_brt(f'scores/VER4_rt_{W}_L0.5{sfx}.json', f'scores/L4_rt_{W}_L0.5{sfx}.json'))
        rep.append(same_brt(f'scores/VER4_{RS}_L0.5{sfx}.json', f'scores/L4_{RS}_L0.5{sfx}.json'))
        rep.append(same_brt(f'scores/VER4_rt_{PICK}_L0.1{sfx}.json', f'scores/R4X_rt_{PICK}_L0.1{sfx}.json'))
        rep.append(same_brt(f'scores/VER4_{RS}_L0.1{sfx}.json', f'scores/B_{RS}_L0.1{sfx}.json'))
    rep.append(same_atk(f'attack/VER4_rt_{PICK}_L0.5.json', f'attack/R4X_rt_{PICK}_L0.5.json'))
    rep.append(same_atk(f'attack/VER4_rt_{W}_L0.5.json', f'attack/L4_rt_{W}_L0.5.json'))
    rep.append(same_atk(f'attack/VER4_{RS}_L0.5.json', f'attack/L4_{RS}_L0.5.json'))
    rep.append(same_atk(f'attack/VER4_rt_{PICK}_L0.1.json', f'attack/R4X_rt_{PICK}_L0.1.json'))
    rep.append(same_atk(f'attack/VER4_{RS}_L0.1.json', f'attack/{RS}_L0.1.json'))
    res['reproduction'] = rep
    res['reproduced_all'] = all(r['identical'] for r in rep)

    # 2. pick = W where the regimes are off --------------------------------------------------------------------
    eq = []
    for v in ('sung', 'sung_low'):
        for lat in ('micbt_cal', 'wired'):
            eq.append(same_live(f'live_bench/VER4_rt_{PICK}_{v}_{lat}.json', f'live_bench/VER4_rt_{W}_{v}_{lat}.json',
                                skip_spec=True))
    for lat in ('micbt_cal_m25', 'micbt_cal_p25'):
        eq.append(same_live(f'live_bench/VER4_rt_{PICK}_sung_{lat}.json', f'live_bench/L4_rt_{W}_sung_{lat}.json',
                            skip_spec=True))
    eq.append(same_live(f'live_bench/VER4_rt_{PICK}_phone_v11.json', f'live_bench/VER4_rt_{W}_phone_v11.json',
                        skip_spec=True))
    for sfx in ('', '_seed2'):
        eq.append(same_brt(f'scores/VER4_rt_{PICK}_L0.1{sfx}.json', f'scores/VER3_rt_{W}_L0.1{sfx}.json', skip_spec=True))
    eq.append(same_atk(f'attack/VER4_rt_{PICK}_L0.1.json', f'attack/VER3_rt_{W}_L0.1.json', skip_spec=True))
    res['pick_equals_W'] = eq

    # 3. the rule from the verifier's numbers -------------------------------------------------------------------
    cells = {n: {f'{v}@{lat}': head(f'live_bench/VER4_rt_{n}_{v}_{lat}.json') for v, lat in CELLS}
             for n in (PICK, W, 'r4_X_hdgsafe_v1', 'r4_X_hgsafe_v1')}
    k1 = {n: sum(c['rt_nd_live'] for c in cc.values()) / 4 for n, cc in cells.items()}
    fs = {n: sum(c['fresh_per100'] + c['stumbles_per100'] for c in cc.values()) for n, cc in cells.items()}
    k2 = {}
    for key, c in cells[PICK].items():
        w = cells[W][key]
        for m, cap in (('harsh', 0.003), ('fresh_per100', 0.5), ('stumbles_per100', 1.0)):
            k2[f'{key} {m} {c[m]:.4f} <= W {w[m]:.4f} + {cap}'] = c[m] <= w[m] + cap + 1e-12
    v11p = head(f'live_bench/VER4_rt_{PICK}_phone_v11.json')
    v11r = head(f'live_bench/VER4_{RS}_phone_v11.json')
    k3 = {f"v11 fresh_heard {v11p['fresh_heard']} <= 4": v11p['fresh_heard'] <= 4,
          f"v11 harsh_heard {v11p['harsh_heard']:.4f} <= 0.1602": v11p['harsh_heard'] <= 0.1602}
    b5 = {n: load(f'scores/VER4_{n}_L0.5.json')['headline'] for n in (f'rt_{PICK}', f'rt_{W}', RS)}
    b5s2 = {n: load(f'scores/VER4_{n}_L0.5_seed2.json')['headline'] for n in (f'rt_{PICK}', f'rt_{W}', RS)}
    f5 = {n: load(f'attack/VER4_{n}_L0.5.json')['headline']['fresh_clash_per100'] for n in (f'rt_{PICK}', f'rt_{W}', RS)}
    p, r = f'rt_{PICK}', RS
    k4 = {f"L0.5 rt_nd {b5[p]['rt_nd']:.4f} >= reactive {b5[r]['rt_nd']:.4f} + 0.005": b5[p]['rt_nd'] >= b5[r]['rt_nd'] + 0.005,
          f"L0.5 harsh {b5[p]['harsh']:.4f} <= reactive {b5[r]['harsh']:.4f}": b5[p]['harsh'] <= b5[r]['harsh'],
          f'L0.5 fresh {f5[p]:.2f} <= reactive {f5[r]:.2f}': f5[p] <= f5[r] + 1e-12,
          f"L0.5 seed2 delta {b5s2[p]['rt_nd'] - b5s2[r]['rt_nd']:+.4f} >= +0.003": b5s2[p]['rt_nd'] - b5s2[r]['rt_nd'] >= 0.003}
    probe = load('verify/VER4_causal_probe_hgeh_v1.json')
    k5 = {'causal_ok': probe['causal_ok'], 'probe_power_ok': probe['probe_power_ok'],
          'new_code_exercised_at_L0.5': probe['new_code_exercised_at_L0.5']}
    kt_p = KT.row((f'scores/VER4_rt_{PICK}_L0.1.json', f'scores/VER4_rt_{PICK}_L0.1_seed2.json',
                   f'attack/VER4_rt_{PICK}_L0.1.json'))
    kt_r = KT.row((f'scores/VER4_{RS}_L0.1.json', f'scores/VER4_{RS}_L0.1_seed2.json', f'attack/VER4_{RS}_L0.1.json'))
    g1 = KT.checks(kt_p, kt_r)
    wp = {v: head(f'live_bench/VER4_rt_{PICK}_{v}_wired.json') for v in ('sung', 'sung_low')}
    ww = {v: head(f'live_bench/VER4_rt_{W}_{v}_wired.json') for v in ('sung', 'sung_low')}
    wr = {v: head(f'live_bench/VER4_{RS}_{v}_wired.json') for v in ('sung', 'sung_low')}
    g2 = {'sung rt_nd >= reactive+0.005': wp['sung']['rt_nd_live'] >= wr['sung']['rt_nd_live'] + 0.005,
          'sung_low rt_nd >= reactive': wp['sung_low']['rt_nd_live'] >= wr['sung_low']['rt_nd_live']}
    for v in ('sung', 'sung_low'):
        g2[f'{v} harsh <= reactive'] = wp[v]['harsh'] <= wr[v]['harsh']
        g2[f'{v} fresh <= reactive'] = wp[v]['fresh_per100'] <= wr[v]['fresh_per100']
    g3 = {'v11 fresh_heard <= reactive': v11p['fresh_heard'] <= v11r['fresh_heard'],
          'v11 harsh_heard <= reactive+0.005': v11p['harsh_heard'] <= v11r['harsh_heard'] + 0.005}
    wm_p = (wp['sung']['rt_nd_live'] + wp['sung_low']['rt_nd_live']) / 2
    wm_w = (ww['sung']['rt_nd_live'] + ww['sung_low']['rt_nd_live']) / 2
    passes = {'K1': k1[PICK] >= k1[W] + 0.005, 'K2': all(k2.values()), 'K3': all(k3.values()), 'K4': all(k4.values()),
              'K5': k5['causal_ok'] and k5['probe_power_ok'] and k5['new_code_exercised_at_L0.5']}
    use_ok = all(g1.values()) and all(g2.values()) and all(g3.values()) and wm_p >= wm_w - 0.003
    tab = load('r4/X/table_v1.json')
    band_rows = {n: {'k1': rr['k1_mean'], 'f_s': rr['k_fresh_stumbles'], 'k1_k4': n in tab['pass_K1_K4']}
                 for n, rr in tab['rows'].items()}
    for n in ('r4_X_hdgsafe_v1', 'r4_X_hgsafe_v1', PICK):
        band_rows[n]['k1_verifier'] = k1[n]
        band_rows[n]['f_s_verifier'] = fs[n]
    passing = [n for n, b in band_rows.items() if b['k1_k4']]
    top = max(band_rows[n].get('k1_verifier', band_rows[n]['k1']) for n in passing)
    band = sorted((n for n in passing if band_rows[n].get('k1_verifier', band_rows[n]['k1']) >= top - 0.003 - 1e-12),
                  key=lambda n: (band_rows[n].get('f_s_verifier', band_rows[n]['f_s']),
                                 -band_rows[n].get('k1_verifier', band_rows[n]['k1'])))
    res['rule'] = {
        'cells': cells, 'k1': k1, 'k1_bar': k1[W] + 0.005, 'k1_delta': k1[PICK] - k1[W], 'fresh_plus_stumbles': fs,
        'K2': k2, 'K3': k3, 'K4': k4, 'K4_numbers': {'rt_nd': {n: b5[n]['rt_nd'] for n in b5},
                                                     'seed2': {n: b5s2[n]['rt_nd'] for n in b5s2},
                                                     'harsh': {n: b5[n]['harsh'] for n in b5}, 'fresh': f5},
        'K5': k5, 'passes': passes, 'pick_passes_K1_K5': all(passes.values()),
        'pool_passing_K1_K4_from_table': passing,
        'band': [(n, band_rows[n].get('k1_verifier', band_rows[n]['k1']),
                  band_rows[n].get('f_s_verifier', band_rows[n]['f_s'])) for n in band],
        'rank_pick': band[0] if band else None,
        'G1': {k: bool(v) for k, v in g1.items()}, 'G1_n': f'{sum(g1.values())}/{len(g1)}',
        'G1_numbers': {'pick': {k: kt_p[k] for k in ('rt_nd', 'harsh', 'seed2', 'fresh', 'us')},
                       'reactive': {k: kt_r[k] for k in ('rt_nd', 'harsh', 'seed2', 'fresh', 'us')}},
        'G2': g2, 'G3': g3, 'wired': {'pick': wp, 'W': ww, 'reactive': wr}, 'wired_mean': {'pick': wm_p, 'W': wm_w},
        'v11': {'pick': v11p, 'reactive': v11r},
        'use': ('single default' if use_ok else 'above 120 ms') if all(passes.values()) else 'none (W everywhere)'}

    # 4. song-level spread -------------------------------------------------------------------------------------
    per = {}
    S = {}
    for v, lat in CELLS:
        a = songs(f'live_bench/VER4_rt_{PICK}_{v}_{lat}.json')
        b = songs(f'live_bench/VER4_rt_{W}_{v}_{lat}.json')
        assert [x[0] for x in a] == [x[0] for x in b] and len(a) == 17
        S[f'{v}@{lat}'] = (a, b)
        d = [x[1] / x[2] - y[1] / y[2] for x, y in zip(a, b)]
        per[f'{v}@{lat}'] = {'wins': sum(x > 1e-12 for x in d), 'losses': sum(x < -1e-12 for x in d),
                             'ties': sum(abs(x) <= 1e-12 for x in d),
                             'per_song_delta': {x[0]: round(dd, 4) for x, dd in zip(a, d)},
                             'fresh_pick_minus_W': sum(x[3] for x in a) - sum(y[3] for y in b),
                             'stumbles_pick_minus_W': round(sum(x[4] for x in a) - sum(y[4] for y in b), 3)}
    both_bt = [sum(1 for k in ('sung@bt_cal', 'sung_low@bt_cal') if per[k]['per_song_delta'][sid] > 0)
               for sid in per['sung@bt_cal']['per_song_delta']]
    rng = random.Random(20261102)
    deltas = []
    for _ in range(10000):
        idx = [rng.randrange(17) for _ in range(17)]
        tot = 0.0
        for key, (a, b) in S.items():
            ha, da = sum(a[i][1] for i in idx), sum(a[i][2] for i in idx)
            hb, db = sum(b[i][1] for i in idx), sum(b[i][2] for i in idx)
            tot += ha / da - hb / db
        deltas.append(tot / 4)
    deltas.sort()
    res['songs'] = {'per_cell': per, 'songs_pick_better_in_both_bt_cal_cells': sum(x == 2 for x in both_bt),
                    'songs_pick_better_in_one_bt_cal_cell': sum(x == 1 for x in both_bt),
                    'songs_pick_better_in_neither_bt_cal_cell': sum(x == 0 for x in both_bt),
                    'bootstrap_k1_delta': {'mean': sum(deltas) / len(deltas), 'p2.5': deltas[249], 'p97.5': deltas[9749],
                                           'P(delta < 0.005)': sum(x < 0.005 for x in deltas) / len(deltas),
                                           'P(delta <= 0)': sum(x <= 0 for x in deltas) / len(deltas)}}

    # 5. robustness -------------------------------------------------------------------------------------------
    rob = {}
    for lat in ROB + ('micbt_cal', 'bt_cal'):
        pk = head(f'live_bench/VER4_rt_{PICK}_sung_{lat}.json')
        wk = head(f'live_bench/L4_rt_{W}_sung_{lat}.json') if lat in ROB else head(f'live_bench/VER4_rt_{W}_sung_{lat}.json')
        rk = head(f'live_bench/VER4_{RS}_sung_{lat}.json') if lat in ROB else head(f'live_bench/L4_{RS}_sung_{lat}.json')
        rob[lat] = {'pick': pk, 'W': wk, 'reactive': rk,
                    'pick_minus_reactive': {m: pk[m] - rk[m] for m in KEYS}, 'pick_minus_W': {m: pk[m] - wk[m] for m in KEYS}}
    low = {}
    for lat in ('bt_cal_m25', 'bt_cal_p25'):
        low[lat] = {'pick': head(f'live_bench/R4X_rt_{PICK}_sung_low_{lat}.json'),
                    'W': head(f'live_bench/L4_rt_{W}_sung_low_{lat}.json')}
    res['robustness_sung'] = rob
    res['robustness_sung_low_report_only'] = low
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1)

    print('reproduced_all', res['reproduced_all'], f"({sum(r['identical'] for r in rep)}/{len(rep)})")
    for r in rep:
        if not r['identical']:
            print('  NOT identical:', r['mine'], r['first_diffs'][:3])
    print('pick == W:', [(e['mine'].split('/')[-1], e['identical'], e['n_diff'], e['first_diffs'][:2]) for e in eq])
    ru = res['rule']
    print(f"K1 pick {k1[PICK]:.4f} W {k1[W]:.4f} bar {k1[W] + 0.005:.4f} delta {k1[PICK] - k1[W]:+.4f}; f+s {fs}")
    print('passes', passes)
    print('K2', k2)
    print('K3', k3, 'K4', k4, 'K5', k5)
    print('band', ru['band'], '-> rank pick', ru['rank_pick'])
    print('G1', ru['G1_n'], 'G2', all(g2.values()), g2, 'G3', g3, f'wired mean pick {wm_p:.4f} W {wm_w:.4f}')
    print('USE:', ru['use'])
    for k, x in per.items():
        print(f"{k}: wins {x['wins']} losses {x['losses']} ties {x['ties']} fresh d {x['fresh_pick_minus_W']} "
              f"stumbles d {x['stumbles_pick_minus_W']}")
    sg = res['songs']
    print('songs better in both bt_cal cells', sg['songs_pick_better_in_both_bt_cal_cells'], 'one',
          sg['songs_pick_better_in_one_bt_cal_cell'], 'neither', sg['songs_pick_better_in_neither_bt_cal_cell'],
          'bootstrap', sg['bootstrap_k1_delta'])
    for lat, x in rob.items():
        print(f"{lat}: pick {x['pick']['rt_nd_live']:.4f}/{x['pick']['harsh']:.4f}/{x['pick']['fresh_per100']:.2f}/"
              f"{x['pick']['stumbles_per100']:.2f}  W {x['W']['rt_nd_live']:.4f}/{x['W']['harsh']:.4f}/"
              f"{x['W']['fresh_per100']:.2f}/{x['W']['stumbles_per100']:.2f}  reactive {x['reactive']['rt_nd_live']:.4f}/"
              f"{x['reactive']['harsh']:.4f}/{x['reactive']['fresh_per100']:.2f}/{x['reactive']['stumbles_per100']:.2f}")
    for lat, x in low.items():
        print(f"sung_low {lat}: pick {x['pick']['rt_nd_live']:.4f} W {x['W']['rt_nd_live']:.4f}")
    print('wrote', out_path)


if __name__ == '__main__':
    main()
