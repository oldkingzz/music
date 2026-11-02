"""Round 4 phase 1 DIAGNOSTIC (run v5rt_1101), not a scored bench: where a live system loses at large latency, read from
its own decision log. Offline only: the _NoSound tripwire is installed before any other import; nothing is played. Val
only (17 val Bach chorales; systems tuned on val; this bundle has no test split). Uses the frozen live bench v2
(futures/v5_rt/live_bench_rt_v2.py: settings) and v1's functions unchanged (make_engine, voice_v1.render,
live_rt_eval_v2.run / voice_notes / attacks / her_at, score); the engine's _record is wrapped only to read each
decision's exact output sample and the note before it (observation only). A fresh system per setting, songs in bench
order (Twinkle, run last by the bench, is skipped), so every setting's metrics must equal the bench file's (checked).

Per latency setting:
 (a) hear attacks (a hear decision that starts a new pitch: switch / enter / reenter; counted like the bench's attacks):
     when they sound after her TRUE onset of the note they answer (ms; tracker matching as live_bench_rt.tracker_rows),
     the share of that note still left, which of her notes they land on (same / a later one / rest), their hit and
     harsh time over the time they sound (her counted singing time) against the note they replaced (the engine's note
     before the decision; -1 = silence) over the same time, fresh clashes, and the stumbles they are part of (as the
     stumbling attack, or as the change that replaced it). The same for plan attacks.
 (b) plan points: the plan decision's note (after a keep: the note Duet keeps) judged over its half beat [q, q + 0.5) of
     her counted singing time, by the state of her note at q (her_at at her grid point q): 'new' (its nominal onset is
     q: it cannot be heard by then), 'held_heard' (began earlier and was answered before this plan decision),
     'held_unheard' (began earlier, not answered yet); also what Duet actually sounded in that window.
 (c) hit time by time since her true onset (0-100, 100-250, 250-450, >= 450 ms): the rt_nd loss against 'wired' split
     into her first 100 ms and the rest.

  .venv/bin/python runs_mini/v5rt_1101/r4/B/diag_lat_v1.py --spec rt:r3_X_regtrkst_v1 --voice sung \
      --lats wired,micbt_cal,bt_cal --out runs_mini/v5rt_1101/r4/B/<new>.json
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline diagnostic: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
OUT_DIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r4', 'B')
LBD = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live_bench')
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import live_bench_rt_v2 as LB2  # noqa: E402  (frozen v2: checks v1's shas, tripwire, round-4 settings)

LB1 = LB2.LB1
BR, EV, VV, np = LB1.BR, LB1.EV, LB1.VV, LB1.np
SR = LB1.SR
J, HARSH = LB1.J, LB1.HARSH
BINS_MS = (0.0, 100.0, 250.0, 450.0, math.inf)
BIN_NAMES = ('0-100', '100-250', '250-450', '450+')
LEFT_NAMES = ('ended', '0-25%', '25-50%', '50-75%', '75-100%')
ATTACKING = ('switch', 'enter', 'reenter')


def left_bin(f):
    if f <= 0:
        return 'ended'
    return LEFT_NAMES[min(4, 1 + int(f * 4 - 1e-9))]


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def new_R():
    return {'bins': {k: [0.0, 0.0, 0.0] for k in BIN_NAMES}, 'hear_rows': [], 'plan_att': [],
            'plan_pts': {k: [0, 0.0, 0.0, 0.0, 0.0, 0.0] for k in ('new', 'held_heard', 'held_unheard')},
            'plan_rest': 0, 'plan_uncounted': 0, 'stumbles': Counter(), 'hear_delay_all': [], 'hear_unmatched': 0}


def one_song(system, spec, song, voice, lat, acc, R):
    T_b = 60.0 / LB1.BPM
    facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LB1.BPM}
    eng = LB1.make_engine(system, spec, facts, LB1.COUNT_IN)
    recs = []
    orig = eng._record

    def rec(kind, at, q, i, r, action, us):
        recs.append({'kind': kind, 'at': int(at), 'q': q, 'i': i, 'action': action,
                     'before': recs[-1]['after'] if recs else -1, 'after': int(eng.live)})
        return orig(kind, at, q, i, r, action, us)

    eng._record = rec
    t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
    y, her, vf = VV.render(song, voice, t0h, T_b)
    x = np.concatenate([np.zeros(int(round(lat['Li'] * SR))), y.astype(np.float64), np.zeros(SR)])
    track, log, _, secs = EV.run(eng, x, Lo=lat['Lo_told'], Li=lat['Li_told'])
    notes, blips = EV.voice_notes(log, lat['Lo'])
    duet = sorted(notes + blips)
    z = [bt['z'] for bt in song['beats']]
    ton = song['tonic']
    LB1.score(her, duet, z, ton, t0h, T_b, acc)                # the bench's own metrics (checked against its file)

    b = lambda s: (s - t0h) / T_b  # noqa: E731
    Ts, H = LB1.step_fn(duet)
    Tb = [b(v) for v in Ts]
    ons = [n[0] for n in her]
    hv = [(on_s, off_s, m, non) for on_s, off_s, m, non, _ in her]
    counted = [non >= J and int(math.floor(non + 1e-9)) < len(z) and z[int(math.floor(non + 1e-9))] >= 0
               for _, _, _, non, _ in her]
    # her counted pieces in beats (score()'s loop): (a0, a1, midi, chord, note index, onset beats)
    hp = []
    for kn, (on_s, off_s, m, non, noff) in enumerate(her):
        on, off = b(on_s), b(off_s)
        for kk in range(max(J, int(math.floor(on))), int(math.floor(off)) + 1):
            a0, a1 = max(on, kk), min(off, kk + 1)
            if a1 - a0 <= 1e-9 or kk >= len(z) or z[kk] < 0:
                continue
            hp.append((a0, a1, m, z[kk], kn, on))
    hp.sort()

    def const(h, xa, xb):
        """Counted time, hit time, harsh time (beats) if Duet sounded h over [xa, xb) (her grid beats)."""
        d = g = bad = 0.0
        for a0, a1, m, zk, _, _ in hp:
            if a0 >= xb:
                break
            w = min(a1, xb) - max(a0, xa)
            if w <= 0:
                continue
            d += w
            g += w * BR.hit(h, m, ton, zk)
            bad += w * (h >= 0 and abs(m - h) % 12 in HARSH)
        return d, g, bad

    # (c) hit time by time since her true onset
    for a0, a1, m, zk, kn, on in hp:
        for x0, x1, h in BR.pieces(Tb, H, a0, a1):
            g = BR.hit(h, m, ton, zk)
            bad = h >= 0 and abs(m - h) % 12 in HARSH
            for j, name in enumerate(BIN_NAMES):
                lo = max(x0, on + BINS_MS[j] / 1000.0 / T_b)
                hi = min(x1, on + BINS_MS[j + 1] / 1000.0 / T_b) if math.isfinite(BINS_MS[j + 1]) else x1
                if hi > lo:
                    R['bins'][name][0] += hi - lo
                    R['bins'][name][1] += (hi - lo) * g
                    R['bins'][name][2] += (hi - lo) * bad

    # heard notes -> her true notes (live_bench_rt.tracker_rows' matching), and when each true note was first answered
    starts = {ii: c0 for c0, ii in eng.by_start.items()}
    heard_true = {}
    for ii in range(len(eng.heard_notes)):
        tau = starts[ii] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + LB1.MATCH_PRE) - 1
        heard_true[ii] = k if (k >= 0 and tau <= her[k][1] + LB1.MATCH_POST) else None
    first_ans = {}
    for idx, r in enumerate(recs):
        if r['kind'] == 'hear':
            k = heard_true.get(r['i'])
            if k is None:
                R['hear_unmatched'] += 1
                continue
            if k not in first_ans:
                first_ans[k] = idx
                if counted[k]:
                    R['hear_delay_all'].append(1000.0 * (r['at'] / SR + lat['Lo'] - her[k][0]))
    by_at = {}
    for idx, r in enumerate(recs):
        by_at.setdefault(r['at'], []).append(idx)

    def rec_near(sample, attacking):
        best = None
        for s in range(sample - 2, sample + 3):
            for idx in by_at.get(s, []):
                if attacking and recs[idx]['action'] not in ATTACKING:
                    continue
                if best is None or idx > best:
                    best = idx
        return best

    # (a) attacks: origin, timing, value against the replaced note, fresh clashes, stumbles
    for xs, h in EV.attacks(duet):
        idx = rec_near(int(round((xs - lat['Lo']) * SR)), True)
        kind = 'none' if idx is None else recs[idx]['kind']
        nt = EV.her_at(xs, hv)
        if nt is None:
            continue
        xb = max(b(xs), nt[3])
        kb = int(math.floor(xb))
        if xb < J or kb >= len(z) or z[kb] < 0:
            continue
        q = bisect.bisect_right(Ts, xs) - 1
        while q + 1 < len(Ts) and H[q + 1] == h:
            q += 1
        t_end = Ts[q + 1] if q + 1 < len(Ts) else xs
        before = -1 if idx is None else recs[idx]['before']
        d, g_new, bad_new = const(h, b(xs), b(t_end))
        _, g_old, bad_old = const(before, b(xs), b(t_end))
        fresh = abs(nt[2] - h) % 12 in HARSH
        k_land = ons.index(nt[0])
        row = {'song': song['id'], 'beat': round(b(xs), 3), 'kind': kind, 'h': int(h), 'before': int(before),
               'd': d, 'hit_new': g_new, 'hit_old': g_old, 'harsh_new': bad_new, 'harsh_old': bad_old,
               'fresh': bool(fresh), 'dur_ms': round(1000 * (t_end - xs), 1)}
        if kind == 'hear':
            r = recs[idx]
            k = heard_true.get(r['i'])
            row['action'] = r['action']
            if k is not None:
                on_s, off_s = her[k][0], her[k][1]
                row['delay_ms'] = round(1000.0 * (xs - on_s), 1)
                row['left'] = max(0.0, (off_s - xs) / (off_s - on_s))
                row['land'] = 'same' if k_land == k else ('later' if k_land > k else 'earlier')
            else:
                row['delay_ms'], row['left'], row['land'] = None, None, 'unmatched'
            R['hear_rows'].append(row)
        else:
            R['plan_att'].append(row)
    # stumbles (live_bench_rt.score's rule) by the origin of the attack and of the change that replaced it
    for xs, h in EV.attacks(duet):
        jn = bisect.bisect_right(ons, xs + LB1.ST_PRE) - 1
        if not (jn >= 0 and counted[jn] and xs <= ons[jn] + LB1.ST_POST):
            continue
        q = bisect.bisect_right(Ts, xs) - 1
        while q + 1 < len(Ts) and H[q + 1] == h:
            q += 1
        t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
        if t_change - xs >= LB1.ST_WIN:
            continue
        ia = rec_near(int(round((xs - lat['Lo']) * SR)), True)
        ir = rec_near(int(round((t_change - lat['Lo']) * SR)), False)
        R['stumbles'][('none' if ia is None else recs[ia]['kind']) + '>' +
                      ('none' if ir is None else recs[ir]['kind'] + ':' + recs[ir]['action'])] += 1

    # (b) plan points
    for idx, r in enumerate(recs):
        if r['kind'] != 'plan':
            continue
        q = r['q']
        nt = EV.her_at(t0h + q * T_b, hv)
        if nt is None:
            R['plan_rest'] += 1
            continue
        kn = ons.index(nt[0])
        if not counted[kn] or q < J:
            R['plan_uncounted'] += 1
            continue
        if abs(her[kn][3] - q) < 1e-6:
            st = 'new'
        else:
            st = 'held_heard' if first_ans.get(kn, math.inf) < idx else 'held_unheard'
        d, g, bad = const(r['after'], q, q + 0.5)
        da, ga, bada = 0.0, 0.0, 0.0
        for a0, a1, m, zk, _, _ in hp:
            if a0 >= q + 0.5:
                break
            lo, hi = max(a0, q), min(a1, q + 0.5)
            if hi <= lo:
                continue
            for x0, x1, h in BR.pieces(Tb, H, lo, hi):
                da += x1 - x0
                ga += (x1 - x0) * BR.hit(h, m, ton, zk)
                bada += (x1 - x0) * (h >= 0 and abs(m - h) % 12 in HARSH)
        P = R['plan_pts'][st]
        P[0] += 1
        P[1] += d
        P[2] += g
        P[3] += bad
        P[4] += ga
        P[5] += bada


def summarize_att(rows, notes, d_all):
    n = len(rows)
    d = sum(r['d'] for r in rows)
    f = lambda a, b: a / b if b else None  # noqa: E731
    out = {'n': n, 'per100': 100.0 * n / notes, 'fresh': int(sum(r['fresh'] for r in rows)),
           'fresh_per100': 100.0 * sum(r['fresh'] for r in rows) / notes,
           'sounding_beats': d, 'hit_share_new': f(sum(r['hit_new'] for r in rows), d),
           'hit_share_replaced': f(sum(r['hit_old'] for r in rows), d),
           'harsh_share_new': f(sum(r['harsh_new'] for r in rows), d),
           'harsh_share_replaced': f(sum(r['harsh_old'] for r in rows), d),
           'net_rt_nd': (sum(r['hit_new'] for r in rows) - sum(r['hit_old'] for r in rows)) / d_all,
           'net_harsh': (sum(r['harsh_new'] for r in rows) - sum(r['harsh_old'] for r in rows)) / d_all,
           'dur_ms_median': pct([r['dur_ms'] for r in rows], 50)}
    return out


def finish(acc, R, notes_n):
    d_all = acc['d']
    hr = R['hear_rows']
    out = {'hear_attacks': summarize_att(hr, notes_n, d_all), 'plan_attacks': summarize_att(R['plan_att'], notes_n, d_all)}
    dl = [r['delay_ms'] for r in hr if r.get('delay_ms') is not None]
    out['hear_attacks'].update({
        'by_action': dict(Counter(r['action'] for r in hr)),
        'delay_ms': {'p10': pct(dl, 10), 'median': pct(dl, 50), 'p90': pct(dl, 90)},
        'left_of_her_note': {k: sum(1 for r in hr if r.get('left') is not None and left_bin(r['left']) == k)
                             for k in LEFT_NAMES},
        'left_median': pct([r['left'] for r in hr if r.get('left') is not None], 50),
        'lands_on': dict(Counter(r['land'] for r in hr)),
        'by_left': {k: summarize_att([r for r in hr if r.get('left') is not None and left_bin(r['left']) == k],
                                     notes_n, d_all) for k in LEFT_NAMES},
        'by_land': {k: summarize_att([r for r in hr if r['land'] == k], notes_n, d_all)
                    for k in sorted(set(r['land'] for r in hr))}})
    out['hear_decisions_delay_ms'] = {'n': len(R['hear_delay_all']), 'p10': pct(R['hear_delay_all'], 10),
                                      'median': pct(R['hear_delay_all'], 50), 'p90': pct(R['hear_delay_all'], 90)}
    out['hear_unmatched'] = R['hear_unmatched']
    st = R['stumbles']
    tot = sum(st.values())
    out['stumbles'] = {'total': tot, 'per100': 100.0 * tot / notes_n, 'by_origin': dict(st.most_common()),
                       'hear_attack_stumbles': sum(v for k, v in st.items() if k.startswith('hear>')),
                       'replaced_by_hear': sum(v for k, v in st.items() if '>hear:' in k)}
    pp = {}
    n_all = sum(P[0] for P in R['plan_pts'].values())
    for k, P in R['plan_pts'].items():
        pp[k] = {'n': P[0], 'share': P[0] / n_all if n_all else None, 'window_beats': P[1],
                 'plan_hit_share': P[2] / P[1] if P[1] else None, 'plan_harsh_share': P[3] / P[1] if P[1] else None,
                 'actual_hit_share': P[4] / P[1] if P[1] else None,
                 'actual_harsh_share': P[5] / P[1] if P[1] else None}
    held = R['plan_pts']['held_heard'][0] + R['plan_pts']['held_unheard'][0]
    out['plan_points'] = {'by_state': pp, 'counted': n_all, 'rest': R['plan_rest'], 'uncounted': R['plan_uncounted'],
                          'not_yet_heard_share': (pp['new']['n'] + pp['held_unheard']['n']) / n_all if n_all else None,
                          'held_unheard_share_of_held': R['plan_pts']['held_unheard'][0] / held if held else None}
    out['bins'] = {k: {'beats': v[0], 'hit_share': v[1] / v[0] if v[0] else None,
                       'harsh_share': v[2] / v[0] if v[0] else None, 'rt_nd_part': v[1] / d_all}
                   for k, v in R['bins'].items()}
    out['bins_check'] = {'sum_beats_minus_d': sum(v[0] for v in R['bins'].values()) - d_all,
                         'sum_hit_minus_hit': sum(v[1] for v in R['bins'].values()) - acc['hit']}
    return out


def bench_file(spec, voice, lat_name):
    n = spec.replace(':', '_')
    for c in (f'L4_{n}_{voice}_{lat_name}.json', f'L4_v2repro_{n}_{voice}_{lat_name}.json',
              f'R3X_{n}_{voice}_{lat_name}.json', f'L_{n}_{voice}_{lat_name}.json'):
        p = os.path.join(LBD, c)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--voice', required=True, choices=VV.VOICES)
    ap.add_argument('--lats', required=True, help='comma-separated live_bench_rt_v2 settings')
    ap.add_argument('--keep_rows', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_DIR + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new .json under {OUT_DIR}/')
    lats = a.lats.split(',')
    for ln in lats:
        if ln not in LB2.LAT:
            sys.exit(f'unknown setting {ln!r}')
    res = {'spec': a.spec, 'voice': a.voice, 'note': 'val only (17 Bach chorales), tuned on val, no test split; '
           'diagnostic, offline, nothing played', 'live_bench_rt_v2_sha256': LB2._sha(LB2.__file__),
           'settings': {}, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    for ln in lats:
        c0 = time.perf_counter()
        lat = LB2.LAT[ln]
        system = LB1.load_system(a.spec)
        acc, R = LB1.new_acc(), new_R()
        for s in LB1.bench3.eval_sets(LB1.SPLIT)['bach']:
            one_song(system, a.spec, s, a.voice, lat, acc, R)
        m = LB1.metrics(acc)
        out = finish(acc, R, m['notes'])
        bf = bench_file(a.spec, a.voice, ln)
        chk = None
        if bf:
            with open(bf) as fh:
                hb = json.load(fh)['headline']
            chk = {'file': os.path.basename(bf), 'same': all(hb[k] == m[k] for k in ('rt_nd_live', 'harsh', 'fresh',
                                                                                       'stumbles_per100', 'notes'))}
        out.update({'latency': lat, 'round_trip_error_ms': round(1000 * LB2.round_trip_error(ln), 1), 'metrics': m,
                    'acc': {k: acc[k] for k in ('d', 'hit', 'harsh', 'd_first', 'hit_first', 'notes')},
                    'bench_check': chk, 'secs': round(time.perf_counter() - c0, 1)})
        if a.keep_rows:
            out['hear_rows'] = R['hear_rows']
        res['settings'][ln] = out
        ha, pa, pp = out['hear_attacks'], out['plan_attacks'], out['plan_points']
        print(f"{a.spec} {a.voice} {ln}: rt_nd_live {m['rt_nd_live']:.4f} (bench file {chk}) | hear attacks "
              f"{ha['per100']:.1f}/100 delay med {ha['delay_ms']['median']} ms, left med {ha['left_median']}, "
              f"lands {ha['lands_on']}, hit new {ha['hit_share_new']:.3f} vs replaced {ha['hit_share_replaced']:.3f}, "
              f"net {ha['net_rt_nd']:+.4f} | plan attacks {pa['per100']:.1f}/100 net {pa['net_rt_nd']:+.4f} | plan "
              f"not-yet-heard {pp['not_yet_heard_share']:.3f}, held unheard {pp['held_unheard_share_of_held']:.3f} | "
              f"stumbles {out['stumbles']['by_origin']} | {out['secs']} s", flush=True)
    if 'wired' in res['settings']:
        w = res['settings']['wired']
        for ln, out in res['settings'].items():
            dw = w['acc']['d']
            out['loss_vs_wired'] = {k: (w['bins'][k]['rt_nd_part'] - out['bins'][k]['rt_nd_part'])
                                    for k in BIN_NAMES}
            out['loss_vs_wired']['total'] = w['metrics']['rt_nd_live'] - out['metrics']['rt_nd_live']
            out['loss_vs_wired']['first_100ms'] = out['loss_vs_wired']['0-100']
            out['loss_vs_wired']['rest'] = out['loss_vs_wired']['total'] - out['loss_vs_wired']['0-100']
            out['loss_vs_wired']['same_counted_time'] = abs(out['acc']['d'] - dw) < 1e-9
            print(ln, 'loss vs wired', {k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in out['loss_vs_wired'].items()}, flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LB1._js)


if __name__ == '__main__':
    main()
