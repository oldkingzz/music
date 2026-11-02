"""Idea B offline evaluator v2 (round 1, run v5rt_1101; val only, tuned on val).

As offline_d1 (one reactive r2_mix_v2 pass per song with a probe at every plan point; plan rules replayed exactly by
merging their decisions into the reactive step function and scoring with bench_rt.score_song and
attack_metric.song_counts), plus a HEAR rule: at each hear point the system may keep the sounding note instead of
r2_mix_v2's choice h_r. The filter's own prev stays h_r in this replay (so the reactive state is unchanged and the
replay stays exact); a scored system may instead tell the filter what Duet really sings.
Also prints calibration: the onset prior by bins, and for plan moves the predicted harsh risk vs what happened.

  python offline_d2.py <grid json> <new out json> [--seed2] [--L 0.1]
"""
import argparse
import bisect
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402

_s = importlib.util.spec_from_file_location('attack_metric_x', os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'tools',
                                                                            'attack_metric.py'))
AM = importlib.util.module_from_spec(_s)
_s.loader.exec_module(AM)
sys.path.insert(0, os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r1', 'B'))
import blib_d1 as BL  # noqa: E402
from systems import m1_filter_v5lo as M1  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

DSN = ('bach', 'folk', 'pop', 'pop909')
GOOD = BL.GOOD
HARSH = BL.HARSH


class Rec:
    def __init__(self, mix, probe, pri):
        self.f, self.probe, self.pri = mix, probe, pri

    def start(self, meta):
        self.f.start(meta)
        self.probe.start(meta)
        self.t = meta['tonic']
        self.tabs, self.hears, self.prevh = [], [], []
        self.last = None

    def plan(self, q, heard):
        self.tabs.append((q, self.probe.table(self.f, q, heard)))
        return None

    def hear(self, i, onset, midi, prev_dur):
        h = int(self.f.step(i, onset, midi, prev_dur))
        k = int(onset)
        pct = self.f.belief(k) @ M1.CTM                     # P(relative pc is a chord tone), as the chooser saw it
        # prior of the heard pitch given her previous heard note (her current note may be misheard)
        if self.last is None:
            pc = 0.5
        else:
            ph = BL.phase_cat(onset)
            pD = self.pri.p_next(self.prevh[-2:] if len(self.prevh) >= 2 else self.prevh, self.t, min(ph, 1))
            d = midi - self.last
            p = float(pD[d + BL.DMAX]) if abs(d) <= BL.DMAX else 0.0
            pc = 0.9 * p / (0.9 * p + 0.1 / 11 * (1 - p) + 1e-12)
        self.prevh.append((onset, None, midi))
        self.last = midi
        self.hears.append((i, h, midi, pct, pc))
        return h


def plan_rule(tab, h0, P):
    p_on, sing, m_c, Phit, Pharsh, hitc, harshc = tab
    wn = p_on * P['LW']
    wh = (1.0 - p_on) * sing * P['WH']
    tot = wn + wh
    if tot <= 1e-12:
        return None, 0.0
    EV = (wn * Phit + wh * hitc) / tot
    EH = (wn * Pharsh + wh * harshc) / tot
    risk = p_on * Pharsh + (1.0 - p_on) * sing * harshc
    val = EV - P['LAM'] * EH
    v_keep = val[h0 - BL.G0] if h0 >= 0 else 0.0
    best, bv, br = None, v_keep, 0.0
    if h0 >= 0 and P.get('SIL', 0) and 0.0 - P.get('KS', 0.0) > bv:
        best, bv = -1, 0.0 - P.get('KS', 0.0)
    if P['RHO'] < 0:
        return best, br
    cand = val - P['MU'] * risk - P['KAP']
    if h0 >= 0 and P.get('WS', 0.0):
        cand = cand - P['WS'] * np.abs(BL.G - h0)
    ok = (risk <= P['RHO']) & (EV >= P['TAU']) & (p_on >= P.get('PMIN', 0.0))
    if P.get('BELOW', 1):
        ok &= BL.G < m_c
    if h0 >= 0:
        ok[h0 - BL.G0] = False
    if ok.any():
        c2 = np.where(ok, cand, -1e9)
        j = int(np.argmax(c2))
        if c2[j] > bv:
            best, br = int(BL.G[j]), float(risk[j])
    return best, br


def hear_rule(info, cur, t, P):
    """Output at a hear point: h_r (switch), or None = keep the sounding note."""
    i, h_r, midi, pct, pc = info
    mode = P.get('HEAR', 'react')
    if mode == 'react' or cur == h_r:
        return h_r
    if cur >= 0:
        iv = abs(midi - cur) % 12
        good_now = iv in GOOD and (cur < midi or not P.get('KBELOW', 1))
        if good_now and pct[(cur - t) % 12] >= pct[(h_r - t) % 12] - P.get('EPSK', 0.0):
            return None                                   # the sounding note is as good as h_r against what she sang
    if P.get('SUSP') is not None and pc < P['SUSP']:
        # her note is probably misheard: do not attack; keep a note that is not harsh against what she sang, else silence
        if cur >= 0 and (abs(midi - cur) % 12) not in HARSH:
            return None
        return -1 if P.get('SUSPSIL', 0) else h_r
    return h_r


def compose(song, tabs, hears, T, H, P, cal):
    true = sorted(song['sop'], key=lambda n: n[0])
    ons = [n[0] for n in true]
    t = song['tonic']
    ev = [(q, 0, tab) for q, tab in tabs if tab is not None] + [(T[j], 1, hears[j]) for j in range(len(T))]
    ev.sort(key=lambda e: (e[0], e[1]))
    T2, H2, cur, nmove, nsil, nkeep = [], [], -1, 0, 0, 0
    for tt, kind, x in ev:
        if kind == 0:
            h, r = plan_rule(x, cur, P)
            if h is None:
                continue
            if h >= 0:
                nmove += 1
                a = bisect.bisect_right(ons, tt + 1e-9) - 1          # her true note sounding at tt
                if a >= 0 and true[a][1] > tt and tt >= B.J:
                    cal['risk'] += r
                    cal['n'] += 1
                    cal['harsh'] += abs(true[a][2] - h) % 12 in HARSH
            else:
                nsil += 1
        else:
            h = hear_rule(x, cur, t, P)
            if h is None:
                nkeep += 1
                continue
        if T2 and T2[-1] == tt:
            H2[-1] = h
        else:
            T2.append(tt)
            H2.append(h)
        cur = h
    return T2, H2, nmove, nsil, nkeep


def score_all(runs, P, L):
    out = {}
    cal = {'risk': 0.0, 'n': 0, 'harsh': 0}
    for ds in DSN:
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        nm = ns = nk = 0
        for song, tabs, hears, T, H in runs[ds]:
            if P is None:
                T2, H2, a, b, c = T, H, 0, 0, 0
            else:
                T2, H2, a, b, c = compose(song, tabs, hears, T, H, P, cal)
            nm += a
            ns += b
            nk += c
            B.score_song(song, T2, H2, L, acc)
            AM.song_counts(song, T2, H2, L, acc2)
        n = acc['notes']
        out[ds] = {'rt_nd': acc['hit'] / acc['d'], 'harsh': acc['harsh'] / acc['d'],
                   'nd_first': acc['hit_first'] / acc['d_first'], 'harsh_first': acc['harsh_first'] / acc['d_first'],
                   'on_time_right': acc['on_time_right'] / n, 'changes': acc['changes'] / n,
                   'late_entry': acc['late_entry'] / n, 'fresh100': 100 * acc2['fresh_clash'] / acc2['notes'],
                   'attacks100': 100 * acc2['attacks'] / acc2['notes'], 'moves100': 100 * nm / n,
                   'sil100': 100 * ns / n, 'keep100': 100 * nk / n}
    head = {k: sum(out[ds][k] for ds in DSN) / 4 for k in out['bach']}
    head['cal_moves'] = cal['n']
    head['cal_risk'] = cal['risk'] / max(cal['n'], 1)
    head['cal_harsh'] = cal['harsh'] / max(cal['n'], 1)
    return head, out


def onset_calibration(runs):
    bins = np.zeros((10, 3))
    for ds in DSN:
        for song, tabs, hears, T, H in runs[ds]:
            ons = sorted(n[0] for n in song['sop'])
            for q, tab in tabs:
                if tab is None:
                    continue
                a = bisect.bisect_left(ons, q)
                y = a < len(ons) and ons[a] < q + BL.ONSET_EPS
                b = min(9, int(tab[0] * 10))
                bins[b] += (1, tab[0], y)
    print('onset prior calibration (val): bin n mean_pred observed', flush=True)
    for b in range(10):
        if bins[b, 0]:
            print(f'  {b / 10:.1f}-{(b + 1) / 10:.1f} n {int(bins[b, 0]):6d} pred {bins[b, 1] / bins[b, 0]:.3f} '
                  f'obs {bins[b, 2] / bins[b, 0]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('out')
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--L', type=float, default=0.1)
    ap.add_argument('--calib', action='store_true')
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid.strip().startswith('[') else json.load(open(a.grid))
    if os.path.exists(a.out):
        sys.exit('refusing to overwrite ' + a.out)
    heard = B.heard_seed2 if a.seed2 else B.HEARD1
    B.bench3.B2.heard = heard
    t0 = time.time()
    train = B.bench3.train_sets()
    pri = BL.Priors(train)
    print(f'priors {time.time() - t0:.1f}s', flush=True)
    rec = Rec(r2_mix_v2.build(train), BL.Probe(pri), pri)
    runs = {}
    t1 = time.time()
    for ds, songs in B.bench3.eval_sets('val').items():
        runs[ds] = []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, up, uh = B.simulate(rec, meta, notes, a.L)
            assert len(T) == len(rec.hears), (len(T), len(rec.hears))
            runs[ds].append((s, rec.tabs, rec.hears, T, H))
    print(f'reactive pass with probes {time.time() - t1:.1f}s', flush=True)
    if a.calib:
        onset_calibration(runs)
    res = []
    h0, r0 = score_all(runs, None, a.L)
    print('reactive  ' + fmt(h0, r0), flush=True)
    res.append({'P': None, 'head': h0, 'rows': r0})
    for P in grid:
        t2 = time.time()
        h, r = score_all(runs, P, a.L)
        res.append({'P': P, 'head': h, 'rows': r})
        print(json.dumps(P) + f' ({time.time() - t2:.0f}s)\n          ' + fmt(h, r, h0, r0), flush=True)
    with open(a.out, 'x') as fh:
        json.dump({'seed': 'seed2' if a.seed2 else 'seed1', 'L': a.L, 'split': 'val', 'results': res}, fh)


def fmt(h, r, h0=None, r0=None):
    s = (f"rt_nd {h['rt_nd']:.4f} harsh {h['harsh']:.4f} fresh {h['fresh100']:.2f} nd1 {h['nd_first']:.3f} "
         f"h1 {h['harsh_first']:.3f} ontime {h['on_time_right']:.3f} chg {h['changes']:.3f} late {h['late_entry']:.3f} "
         f"att {h['attacks100']:.1f} moves {h['moves100']:.1f} sil {h['sil100']:.1f} keep {h['keep100']:.1f} "
         f"cal {h.get('cal_risk', 0):.3f}/{h.get('cal_harsh', 0):.3f} |")
    for ds in DSN:
        x = r[ds]
        s += f" {ds} {x['rt_nd']:.4f}/{x['harsh']:.4f}/{x['fresh100']:.2f}"
    if h0 is not None:
        s += (f"\n          delta rt_nd {h['rt_nd'] - h0['rt_nd']:+.4f} harsh {h['harsh'] - h0['harsh']:+.4f} fresh "
              f"{h['fresh100'] - h0['fresh100']:+.2f} | " + ' '.join(
                  f"{ds} {r[ds]['rt_nd'] - r0[ds]['rt_nd']:+.4f}/{r[ds]['harsh'] - r0[ds]['harsh']:+.4f}/"
                  f"{r[ds]['fresh100'] - r0[ds]['fresh100']:+.2f}" for ds in DSN))
    return s


if __name__ == '__main__':
    main()
