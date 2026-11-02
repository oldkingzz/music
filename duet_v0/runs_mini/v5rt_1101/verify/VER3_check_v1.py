"""VER3_check_v1 (verifier, run v5rt_1101, round 3): reproduction, gates and robustness for the round-3 winner
rt:r3_X_regtrkst_v1 vs reactive:r2_mix_v2, read from the VER3_* files this verifier wrote and from the files the round-3
agents reported. Val only (bench_rt: the 4 val datasets; live bench: the 17 val Bach chorales); every system was tuned
on val; this bundle has no test split. Read-only on its inputs; writes ONE new JSON (mode 'x') under
runs_mini/v5rt_1101/verify/.

1. reproduction: every VER3_* file vs the reported one, exact equality of every field except timings (us_*, time,
   timing); the live files also by digest.
2. gates from the VER3_* numbers: G1 (keep_table_v1.row / checks, copied), G2 and G3 (LB_select_v1's checks, copied).
3. robustness: bt, clean, phone_twinkle, bench_rt L 0.05 / 0.15, the live probe's tempo and block-phase conditions;
   song-level spread on sung / sung_low (wins, song bootstrap of the pooled delta).
4. register sanity: Duet's MIDI range and the share of her counted singing time with Duet ABOVE her true note (sung_low
   and sung, both systems; her true notes from voice_v1.true_notes exactly as live_bench_rt builds them); v11: share of
   answered notes with Duet above the heard pitch.
5. tracker table provenance: models/E_tracker_seq_v1.json is TRAIN only (no val id) and models/E_trk_tab_v1.npz equals
   a refit from it with runs_mini/v5rt_1101/r3/E/fit_trk_tab_E_v1.py's own functions (in memory; nothing written).

  .venv/bin/python runs_mini/v5rt_1101/verify/VER3_check_v1.py --out runs_mini/v5rt_1101/verify/VER3_check_result_v1.json
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
OUT_DIR = os.path.join(RUN, 'verify')
BR = LBR.BR
DS = ('bach', 'folk', 'pop', 'pop909')
TIMING = {'us_mean', 'us_max', 'us_p99', 'us_mean_plan', 'us_mean_hear', 'us_total', 'time', 'timing'}
W, R = 'rt_r3_X_regtrkst_v1', 'reactive_r2_mix_v2'
FRESH_MAX = 1.8100634866373206


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def diff(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in TIMING:
                continue
            if k not in a or k not in b:
                out.append((path + '/' + k, 'missing' if k not in a else 'present', 'missing' if k not in b else 'present'))
                continue
            out += diff(a[k], b[k], path + '/' + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f'len {len(a)}', f'len {len(b)}'))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f'{path}[{i}]')
    elif a != b:
        out.append((path, a, b))
    return out


def compare(mine, theirs, keys=None):
    a, b = load(mine), load(theirs)
    if keys:
        a = {k: a[k] for k in keys}
        b = {k: b[k] for k in keys}
    d = diff(a, b)
    return {'mine': mine, 'reported': theirs, 'identical': not d, 'n_diff': len(d), 'first_diffs': [list(map(str, x))
                                                                                                   for x in d[:5]]}


# ---------------------------------------------------------------------------------------------------- gates
def g1_row(paths):                                     # keep_table_v1.row (copied)
    s, s2, a = (load(p) for p in paths)
    assert s['L'] == 0.1 and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and s['split'] == 'val', paths
    assert a['L'] == 0.1 and a['seed'] == 'seed1', paths
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'us': s['headline']['us_mean'], 'fresh': a['headline']['fresh_clash_per100'],
            'ds_rt_nd': {d: s['rows'][d]['rt_nd'] for d in DS}, 'ds_harsh': {d: s['rows'][d]['harsh'] for d in DS},
            'ds_fresh': {d: a['rows'][d]['fresh_clash_per100'] for d in DS}, 'bench_sha256': s['bench_sha256']}


def g1_checks(r, b):                                   # keep_table_v1.checks (copied)
    c = {'rt_nd +0.005': r['rt_nd'] >= b['rt_nd'] + 0.005, 'harsh <= reactive': r['harsh'] <= b['harsh'],
         'seed2 +0.003': r['seed2'] - b['seed2'] >= 0.003, 'us < 2000': r['us'] < 2000,
         'fresh <= reactive': r['fresh'] <= b['fresh'] + 1e-12, 'fresh <= 1.8100634866373206': r['fresh'] <= FRESH_MAX}
    for d in DS:
        c[f'{d} harsh <= +0.005'] = r['ds_harsh'][d] <= b['ds_harsh'][d] + 0.005
        c[f'{d} rt_nd >= -0.01'] = r['ds_rt_nd'][d] >= b['ds_rt_nd'][d] - 0.01
        c[f'{d} fresh <= +1.0'] = r['ds_fresh'][d] <= b['ds_fresh'][d] + 1.0
    return c


def g2_checks(s, sl, rs, rsl):                         # LB_select_v1's G2 (copied)
    return {'sung rt_nd_live >= reactive + 0.005': s['rt_nd_live'] >= rs['rt_nd_live'] + 0.005,
            'sung_low rt_nd_live >= reactive': sl['rt_nd_live'] >= rsl['rt_nd_live'],
            'sung harsh <= reactive': s['harsh'] <= rs['harsh'],
            'sung_low harsh <= reactive': sl['harsh'] <= rsl['harsh'],
            'sung fresh_per100 <= reactive': s['fresh_per100'] <= rs['fresh_per100'],
            'sung_low fresh_per100 <= reactive': sl['fresh_per100'] <= rsl['fresh_per100']}


def g3_checks(v, rv):                                  # LB_select_v1's G3 (copied)
    return {'v11 fresh_heard <= reactive': v['fresh_heard'] <= rv['fresh_heard'],
            'v11 harsh_heard <= reactive + 0.005': v['harsh_heard'] <= rv['harsh_heard'] + 0.005}


# ---------------------------------------------------------------------------------------------------- song spread
def spread(fw, fr, n_boot=10000, seed=20261102):
    a, b = load(fw)['songs'], load(fr)['songs']
    assert [s['id'] for s in a] == [s['id'] for s in b]
    d = np.array([s['metrics']['counted_beats'] for s in a])
    assert np.array_equal(d, np.array([s['metrics']['counted_beats'] for s in b]))
    hw = np.array([s['metrics']['rt_nd_live'] for s in a]) * d
    hr = np.array([s['metrics']['rt_nd_live'] for s in b]) * d
    dl = hw / d - hr / d
    rng = random.Random(seed)
    boots = []
    n = len(a)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(hw[idx].sum() / d[idx].sum() - hr[idx].sum() / d[idx].sum())
    boots = np.array(boots)
    fw_, fr_ = [s['metrics']['fresh'] for s in a], [s['metrics']['fresh'] for s in b]
    return {'songs': n, 'wins': int((dl > 1e-12).sum()), 'losses': int((dl < -1e-12).sum()),
            'ties': int((np.abs(dl) <= 1e-12).sum()), 'pooled_delta': float(hw.sum() / d.sum() - hr.sum() / d.sum()),
            'median_song_delta': float(np.median(dl)), 'min_song_delta': float(dl.min()),
            'max_song_delta': float(dl.max()), 'boot_ci95': [float(np.percentile(boots, 2.5)),
                                                             float(np.percentile(boots, 97.5))],
            'boot_p_delta_ge_0.005': float((boots >= 0.005).mean()), 'boot_p_delta_ge_0': float((boots >= 0).mean()),
            'fresh_songs_more_than_reactive': int(sum(x > y for x, y in zip(fw_, fr_))),
            'fresh_songs_fewer_than_reactive': int(sum(x < y for x, y in zip(fw_, fr_))),
            'per_song': [[s['id'], round(float(x), 4)] for s, x in zip(a, dl)]}


# ---------------------------------------------------------------------------------------------------- register
def register(fname, voice):
    """Duet's MIDI range and time above her true note, over her counted singing time (the rt_nd_live denominator)."""
    res = load(fname)
    assert res['voice'] == voice and res['lat'] == 'wired'
    T_b = 60.0 / LBR.BPM
    t0h = 0.5 + LBR.COUNT_IN * T_b                     # RTDuet's t0 (lead 0.5); wired: her grid = the engine's
    shift = -12 if voice == 'sung_low' else 0
    songs = {s['id']: s for s in LBR.bench3.eval_sets('val')['bach']}
    tot = defaultdict(float)
    mids, hers, att_above, att_n = [], [], 0, 0
    max_on_err = 0.0
    for r in res['songs']:
        song = songs[r['id']]
        true = LBR.VV.true_notes(song, t0h, T_b, shift)
        her = [((a - t0h) / T_b, (e - t0h) / T_b, m, non, noff) for a, e, m, non, noff in true]
        for (on, _, m, non, _), row in zip(her, r['her']):
            max_on_err = max(max_on_err, abs(round(on, 4) - row[1]))
            assert row[0] == non and row[2] == m
        duet = [(float(a), float(e), int(m)) for a, e, m in r['duet']]
        mids += [m for a, e, m in duet if e > a]
        hers += [m for _, _, m, _, _ in her]
        Ts, H = LBR.step_fn(duet)
        z = [b['z'] for b in song['beats']]
        ton = song['tonic']
        d_song = 0.0
        for on, off, m, non, noff in her:
            for k in range(max(LBR.J, int(math.floor(on))), int(math.floor(off)) + 1):
                ov = min(off, k + 1) - max(on, k)
                if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                    continue
                tot['d'] += ov
                d_song += ov
                for x0, x1, h in BR.pieces(Ts, H, max(on, k), min(off, k + 1)):
                    w = x1 - x0
                    tot['sounding'] += w * (h >= 0)
                    tot['above'] += w * (h > m)
                    tot['above_oct'] += w * (h >= m + 12)
                    tot['below'] += w * (0 <= h < m)
                    tot['unison'] += w * (h == m)
                    tot['hit'] += w * BR.hit(h, m, ton, z[k])
                    tot['hit_above'] += w * (h > m and BR.hit(h, m, ton, z[k]))
        assert abs(d_song - r['metrics']['counted_beats']) < 1e-6, (r['id'], d_song, r['metrics']['counted_beats'])
        hv = [(a * T_b + t0h, e * T_b + t0h, m, non) for a, e, m, non, _ in her]
        for x, h in LBR.EV.attacks([(a * T_b + t0h, e * T_b + t0h, m) for a, e, m in duet]):
            nt = LBR.EV.her_at(x, hv)
            if nt is not None:
                att_n += 1
                att_above += h > nt[2]
    d = tot['d']
    return {'file': fname, 'duet_midi_min': int(min(mids)), 'duet_midi_max': int(max(mids)),
            'duet_midi_p05_p50_p95': [float(np.percentile(mids, p)) for p in (5, 50, 95)],
            'her_midi_min': int(min(hers)), 'her_midi_max': int(max(hers)),
            'share_of_her_time_duet_sounding': tot['sounding'] / d, 'share_of_her_time_duet_above': tot['above'] / d,
            'share_of_her_time_duet_octave_plus_above': tot['above_oct'] / d,
            'share_of_her_time_duet_below': tot['below'] / d, 'share_of_her_time_unison': tot['unison'] / d,
            'share_of_duet_sounding_time_above_her': tot['above'] / tot['sounding'] if tot['sounding'] else None,
            'hit_share_recomputed_from_rounded_notes': tot['hit'] / d, 'reported_rt_nd_live': res['headline']['rt_nd_live'],
            'hit_time_above_her_share_of_hits': tot['hit_above'] / tot['hit'] if tot['hit'] else None,
            'attacks_on_her': att_n, 'attacks_above_her': att_above,
            'max_onset_check_err_beats': max_on_err}


def v11_above(fname):
    p = load(fname)['phone']
    ans = p['answered']
    n = len(ans)
    snd = sum(1 for a in ans if a[2] >= 0)
    above = sum(1 for a in ans if a[2] >= 0 and a[2] > a[1])
    mids = [m for _, _, m in p['duet_s']]
    return {'answered': n, 'duet_sounding_after_answer': snd, 'duet_above_heard_after_answer': above,
            'duet_midi_min': min(mids) if mids else None, 'duet_midi_max': max(mids) if mids else None,
            'her_heard_min': min(a[1] for a in ans), 'her_heard_max': max(a[1] for a in ans)}


# ---------------------------------------------------------------------------------------------------- tracker table
def trk_tab_check():
    sys.path.insert(0, os.path.join(RUN, 'r3', 'E'))
    import fit_trk_tab_E_v1 as FT  # noqa: E402  (its main() is not run)
    with open(FT.SEQ) as fh:
        d = json.load(fh)
    tr_ids = {s['id'] for s in LBR.bench3.train_sets()['bach']}
    val = LBR.bench3.eval_sets('val')
    val_ids = {str(s.get('id', s.get('title', ''))) for ds in val for s in val[ds]}
    seq_ids = set(d['song_ids'])
    row_ids = {r[0] for r in d['rows']}
    ix = {k: j for j, k in enumerate(d['fields'])}
    seqs = defaultdict(list)
    for r in d['rows']:
        seqs[(r[0], r[1])].append(r)
    EVv = FT.EV
    N3 = np.zeros((11, 12, 2, len(EVv)))
    NF3 = np.zeros((11, 12, 2))
    F = np.zeros(len(EVv))
    for key, rows in seqs.items():
        rows.sort(key=lambda r: r[ix['i']])
        seq = [(r[ix['onset']], r[ix['heard']]) for r in rows]
        for j, r in enumerate(rows):
            if r[ix['k']] < 0:
                continue
            e = r[ix['true_midi']] - r[ix['heard']]
            if abs(e) > 6:
                continue
            fz = FT.feats(seq, j)
            if fz is None:
                F[e + 6] += 1
                continue
            c1b, gc, ret = fz
            N3[c1b, gc, ret, e + 6] += 1
            NF3[c1b, gc, ret] += bool(r[ix['frag']])
    A = FT.A
    N2, N1, N0 = N3.sum(2), N3.sum((1, 2)), N3.sum((0, 1, 2))
    P0 = (N0 + 1.0) / (N0.sum() + len(EVv))
    P1 = (N1 + A * P0[None]) / (N1.sum(-1, keepdims=True) + A)
    P2 = (N2 + A * P1[:, None]) / (N2.sum(-1, keepdims=True) + A)
    P3 = (N3 + A * P2[:, :, None]) / (N3.sum(-1, keepdims=True) + A)
    n3, n2, n1 = N3.sum(-1), N2.sum(-1), N1.sum(-1)
    f0 = NF3.sum() / n3.sum()
    f1 = (NF3.sum((1, 2)) + A * f0) / (n1 + A)
    f2 = (NF3.sum(2) + A * f1[:, None]) / (n2 + A)
    PF3 = (NF3 + A * f2[:, :, None]) / (n3 + A)
    Fp = (F + 1.0) / (F.sum() + len(EVv))
    z = np.load(os.path.join(RUN, 'models', 'E_trk_tab_v1.npz'))
    eq = {k: float(np.max(np.abs(z[k] - v))) for k, v in (('P3', P3), ('PF3', PF3), ('F', Fp), ('N3', N3), ('NF3', NF3))}
    return {'seq_split': d['split'], 'seq_songs': d['songs'], 'seq_song_ids': len(seq_ids),
            'seq_ids_all_train': seq_ids <= tr_ids and row_ids <= tr_ids, 'train_bach_ids': len(tr_ids),
            'seq_ids_in_val': sorted(seq_ids & val_ids), 'voices': sorted({r[1] for r in d['rows']}),
            'seq_code_sha256': d['code_sha256'], 'refit_max_abs_diff': eq,
            'refit_identical': all(v == 0.0 for v in eq.values())}


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    path = os.path.abspath(ap.parse_args().out)
    if not path.startswith(OUT_DIR + os.sep) or not os.path.basename(path).startswith('VER3_') \
            or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new VER3_*.json under {OUT_DIR}')
    out = {'note': 'val only; every system tuned on val; this bundle has no test split', 'winner': 'rt:r3_X_regtrkst_v1'}
    rep = {}
    for k in ('L0.1', 'L0.1_seed2'):
        rep[f'bench_win_{k}'] = compare(f'scores/VER3_{W}_{k}.json', f'scores/R3X_{W}_{k}.json',
                                        ('spec', 'L', 'seed', 'split', 'headline', 'rows', 'bench_sha256'))
        rep[f'bench_react_{k}'] = compare(f'scores/VER3_{R}_{k}.json', f'scores/B_{R}_{k}.json',
                                          ('spec', 'L', 'seed', 'split', 'headline', 'rows', 'bench_sha256'))
    for k in ('L0.05', 'L0.15'):
        rep[f'bench_react_{k}'] = compare(f'scores/VER3_{R}_{k}.json', f'scores/B_{R}_{k}.json',
                                          ('spec', 'L', 'seed', 'split', 'headline', 'rows', 'bench_sha256'))
    rep['attack_win'] = compare(f'attack/VER3_{W}_L0.1.json', f'attack/R3X_{W}_L0.1.json',
                                ('spec', 'L', 'seed', 'split', 'headline', 'rows'))
    rep['attack_react'] = compare(f'attack/VER3_{R}_L0.1.json', f'attack/{R}_L0.1.json',
                                  ('spec', 'L', 'seed', 'split', 'headline', 'rows'))
    for v in ('sung_wired', 'sung_low_wired', 'phone_v11', 'sung_bt', 'clean_wired', 'phone_twinkle'):
        a, b = f'live_bench/VER3_{W}_{v}.json', f'live_bench/R3X_{W}_{v}.json'
        c = compare(a, b)
        c['digest_mine'], c['digest_reported'] = load(a)['digest'], load(b)['digest']
        rep[f'live_win_{v}'] = c
        a, b = f'live_bench/VER3_{R}_{v}.json', f'live_bench/L_{R}_{v}.json'
        c = compare(a, b)
        c['digest_mine'], c['digest_reported'] = load(a)['digest'], load(b)['digest']
        rep[f'live_react_{v}'] = c
    out['reproduction'] = rep
    out['reproduced_all'] = all(v['identical'] for v in rep.values())

    # gates from VER3 numbers
    gw = g1_row((f'scores/VER3_{W}_L0.1.json', f'scores/VER3_{W}_L0.1_seed2.json', f'attack/VER3_{W}_L0.1.json'))
    gr = g1_row((f'scores/VER3_{R}_L0.1.json', f'scores/VER3_{R}_L0.1_seed2.json', f'attack/VER3_{R}_L0.1.json'))
    c1 = g1_checks(gw, gr)
    lw = {v: load(f'live_bench/VER3_{W}_{v}_wired.json')['headline'] for v in ('sung', 'sung_low')}
    lr = {v: load(f'live_bench/VER3_{R}_{v}_wired.json')['headline'] for v in ('sung', 'sung_low')}
    vw = load(f'live_bench/VER3_{W}_phone_v11.json')['phone']['proxies']
    vr = load(f'live_bench/VER3_{R}_phone_v11.json')['phone']['proxies']
    c2 = g2_checks(lw['sung'], lw['sung_low'], lr['sung'], lr['sung_low'])
    c3 = g3_checks(vw, vr)
    keys = ('rt_nd_live', 'harsh', 'fresh_per100', 'fresh', 'stumbles_per100', 'on_time_right', 'wrong_att_per100',
            'late_entry', 'first_100ms_hit', 'first_100ms_harsh', 'double', 'notes')
    out['gates'] = {
        'G1': all(c1.values()), 'G2': all(c2.values()), 'G3': all(c3.values()),
        'g1_checks': c1, 'g2_checks': c2, 'g3_checks': c3,
        'g1_winner': gw, 'g1_reactive': gr, 'seed2_delta': gw['seed2'] - gr['seed2'],
        'live_winner': {v: {k: lw[v][k] for k in keys} for v in lw},
        'live_reactive': {v: {k: lr[v][k] for k in keys} for v in lr},
        'v11_winner': vw, 'v11_reactive': vr,
        'mean_live_winner': (lw['sung']['rt_nd_live'] + lw['sung_low']['rt_nd_live']) / 2,
        'mean_live_reactive': (lr['sung']['rt_nd_live'] + lr['sung_low']['rt_nd_live']) / 2,
        'fresh_live_sum_winner': lw['sung']['fresh'] + lw['sung_low']['fresh'],
        'margins': {'bench_rt_nd': gw['rt_nd'] - gr['rt_nd'] - 0.005,
                    'sung_rt_nd_live': lw['sung']['rt_nd_live'] - lr['sung']['rt_nd_live'] - 0.005,
                    'sung_low_rt_nd_live': lw['sung_low']['rt_nd_live'] - lr['sung_low']['rt_nd_live'],
                    'v11_fresh_heard': vr['fresh_heard'] - vw['fresh_heard'],
                    'v11_harsh_heard': vr['harsh_heard'] + 0.005 - vw['harsh_heard']}}

    # robustness
    rob = {}
    for v in ('sung_bt', 'clean_wired'):
        a = load(f'live_bench/VER3_{W}_{v}.json')
        b = load(f'live_bench/VER3_{R}_{v}.json')
        rob[v] = {'winner': {k: a['headline'][k] for k in keys}, 'reactive': {k: b['headline'][k] for k in keys},
                  'twinkle_winner': a['twinkle']['rt_nd_live'], 'twinkle_reactive': b['twinkle']['rt_nd_live']}
    for v in ('sung_wired', 'sung_low_wired'):
        a = load(f'live_bench/VER3_{W}_{v}.json')
        b = load(f'live_bench/VER3_{R}_{v}.json')
        rob[f'twinkle_rendered_{v}'] = {'winner': a['twinkle'], 'reactive': b['twinkle']}
    a = load(f'live_bench/VER3_{W}_phone_twinkle.json')['phone']
    b = load(f'live_bench/VER3_{R}_phone_twinkle.json')['phone']
    rob['phone_twinkle'] = {'winner': dict(a['metrics'], sampled_fit=a['sampled']['fit'], proxies=a['proxies']),
                            'reactive': dict(b['metrics'], sampled_fit=b['sampled']['fit'], proxies=b['proxies'])}
    for L in ('L0.05', 'L0.1', 'L0.15'):
        a, b = load(f'scores/VER3_{W}_{L}.json'), load(f'scores/VER3_{R}_{L}.json')
        ha, hb = a['headline'], b['headline']
        rob[f'bench_{L}'] = {
            'winner': {'rt_nd': ha['rt_nd'], 'harsh': ha['harsh'], 'ds_rt_nd': {d: a['rows'][d]['rt_nd'] for d in DS},
                       'ds_harsh': {d: a['rows'][d]['harsh'] for d in DS}},
            'reactive': {'rt_nd': hb['rt_nd'], 'harsh': hb['harsh'], 'ds_rt_nd': {d: b['rows'][d]['rt_nd'] for d in DS},
                         'ds_harsh': {d: b['rows'][d]['harsh'] for d in DS}},
            'delta': ha['rt_nd'] - hb['rt_nd'],
            'keep_like': {'rt_nd +0.005': ha['rt_nd'] >= hb['rt_nd'] + 0.005, 'harsh <=': ha['harsh'] <= hb['harsh'],
                          'no ds rt_nd < -0.01': all(a['rows'][d]['rt_nd'] >= b['rows'][d]['rt_nd'] - 0.01 for d in DS),
                          'no ds harsh > +0.005': all(a['rows'][d]['harsh'] <= b['rows'][d]['harsh'] + 0.005
                                                      for d in DS)}}
    pw = load('verify/VER3_live_probe_rt_r3_X_regtrkst_v1.json')['conditions']
    pr = load('verify/VER3_live_probe_reactive_r2_mix_v2.json')['conditions']
    probe = {}
    for cond in ('base', 'bpm72', 'bpm110', 'phase'):
        s, sl, rs, rsl = (pw[f'{cond}_sung']['headline'], pw[f'{cond}_sung_low']['headline'],
                          pr[f'{cond}_sung']['headline'], pr[f'{cond}_sung_low']['headline'])
        probe[cond] = {'winner_sung': {k: s[k] for k in keys}, 'winner_sung_low': {k: sl[k] for k in keys},
                       'reactive_sung': {k: rs[k] for k in keys}, 'reactive_sung_low': {k: rsl[k] for k in keys},
                       'g2_like': g2_checks(s, sl, rs, rsl)}
    probe['base_equals_VER3_files'] = {
        'winner': diff(pw['base_sung']['headline'], lw['sung']) == [] and diff(pw['base_sung_low']['headline'],
                                                                                lw['sung_low']) == [],
        'reactive': diff(pr['base_sung']['headline'], lr['sung']) == [] and diff(pr['base_sung_low']['headline'],
                                                                                  lr['sung_low']) == []}
    probe['reg_shift_songs_nonzero'] = {k: sum(1 for x in pw[k]['reg'] if any(e[1] != 0 for e in x['s']))
                                        for k in ('base_sung', 'base_sung_low')}
    probe['reg_shift_values'] = {k: sorted({e[1] for x in pw[k]['reg'] for e in x['s']}) for k in ('base_sung',
                                                                                                    'base_sung_low')}
    probe['reg_resync_songs'] = {k: sum(1 for x in pw[k]['reg'] if x['resync'] > 0) for k in ('base_sung',
                                                                                               'base_sung_low')}
    probe['winner_base_stats'] = {k: pw[k]['base_stats'] for k in ('base_sung', 'base_sung_low')}
    rob['live_probe'] = probe
    rob['song_spread'] = {'sung': spread(f'live_bench/VER3_{W}_sung_wired.json', f'live_bench/VER3_{R}_sung_wired.json'),
                          'sung_low': spread(f'live_bench/VER3_{W}_sung_low_wired.json',
                                             f'live_bench/VER3_{R}_sung_low_wired.json')}
    out['robustness'] = rob
    out['register'] = {f'{who}_{v}': register(f'live_bench/VER3_{f}_{v}_wired.json', v)
                       for who, f in (('winner', W), ('reactive', R)) for v in ('sung_low', 'sung')}
    out['register']['v11_winner'] = v11_above(f'live_bench/VER3_{W}_phone_v11.json')
    out['register']['v11_reactive'] = v11_above(f'live_bench/VER3_{R}_phone_v11.json')
    out['tracker_table'] = trk_tab_check()
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('reproduced_all', out['reproduced_all'])
    for k, v in rep.items():
        print(f"  {k}: identical {v['identical']} n_diff {v['n_diff']} {v.get('digest_mine', '')[:12]} "
              f"{v.get('digest_reported', '')[:12]} {v['first_diffs'][:2]}")
    g = out['gates']
    print('G1', g['G1'], 'G2', g['G2'], 'G3', g['G3'], 'margins', json.dumps(g['margins']))
    print('failed checks', [k for k, v in {**g['g1_checks'], **g['g2_checks'], **g['g3_checks']}.items() if not v])
    print('robust', json.dumps({k: v for k, v in rob.items() if k.startswith('bench')}, default=str)[:1500])
    print('probe g2_like', {c: all(probe[c]['g2_like'].values()) for c in ('base', 'bpm72', 'bpm110', 'phase')},
          'base_equal', probe['base_equals_VER3_files'], 'reg', probe['reg_shift_songs_nonzero'],
          probe['reg_shift_values'], probe['reg_resync_songs'])
    for k, v in rob['song_spread'].items():
        print('spread', k, {x: y for x, y in v.items() if x != 'per_song'})
    for k, v in out['register'].items():
        print('register', k, json.dumps(v, default=str)[:700])
    print('tracker table', json.dumps(out['tracker_table'], default=str)[:900])
    print('wrote', path)


if __name__ == '__main__':
    main()
