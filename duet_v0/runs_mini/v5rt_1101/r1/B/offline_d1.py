"""Idea B offline evaluator (round 1, run v5rt_1101; val only, tuned on val).

One reactive r2_mix_v2 pass through bench_rt.simulate per song, with a probe at every plan point that records the
candidate table (blib_d1.Probe) and never changes Duet's note. Because hear() always answers with r2_mix_v2's own
step() (the GRU and filter state never depend on plan decisions), any plan-point rule can then be replayed exactly:
its decisions are merged into the reactive step function (plan before hear at equal times, as bench_rt does) and
scored with bench_rt.score_song and attack_metric.song_counts, unchanged.

  python offline_d1.py <grid json> <new out json> [--seed2] [--L 0.1]
"""
import argparse
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
from systems import r2_mix_v2  # noqa: E402

DSN = ('bach', 'folk', 'pop', 'pop909')


class Rec:
    def __init__(self, mix, probe):
        self.f, self.probe = mix, probe

    def start(self, meta):
        self.f.start(meta)
        self.probe.start(meta)
        self.tabs = []

    def plan(self, q, heard):
        self.tabs.append((q, self.probe.table(self.f, q, heard)))
        return None

    def hear(self, i, onset, midi, prev_dur):
        return int(self.f.step(i, onset, midi, prev_dur))


def rule(tab, h0, P):
    p_on, sing, m_c, Phit, Pharsh, hitc, harshc = tab
    if p_on < P.get('PMIN', 0.0) and not P.get('HOLDMOVES', 1):
        return None
    wn = p_on * P['LW']
    wh = (1.0 - p_on) * sing * P['WH']
    tot = wn + wh
    if tot <= 1e-12:
        return None
    EV = (wn * Phit + wh * hitc) / tot
    EH = (wn * Pharsh + wh * harshc) / tot
    risk = p_on * Pharsh + (1.0 - p_on) * sing * harshc
    val = EV - P['LAM'] * EH
    v_keep = val[h0 - BL.G0] if h0 >= 0 else 0.0
    best, bv = None, v_keep
    if h0 >= 0 and P.get('SIL', 1) and 0.0 - P.get('KS', 0.0) > bv:
        best, bv = -1, 0.0 - P.get('KS', 0.0)
    cand = val - P['MU'] * risk - P['KAP']
    if h0 >= 0 and P.get('WS', 0.0):
        cand = cand - P['WS'] * np.abs(BL.G - h0)
    ok = (risk <= P['RHO']) & (EV >= P['TAU'])
    if P.get('BELOW', 1):
        ok &= BL.G < m_c
    if h0 >= 0:
        ok[h0 - BL.G0] = False
    if ok.any():
        c2 = np.where(ok, cand, -1e9)
        j = int(np.argmax(c2))
        if c2[j] > bv:
            best = int(BL.G[j])
    return best


def compose(tabs, T, H, P):
    ev = [(q, 0, tab) for q, tab in tabs if tab is not None] + [(t, 1, h) for t, h in zip(T, H)]
    ev.sort(key=lambda e: (e[0], e[1]))
    T2, H2, cur, nmove, nsil = [], [], -1, 0, 0
    for t, kind, x in ev:
        if kind == 0:
            h = rule(x, cur, P)
            if h is None:
                continue
            nmove += h >= 0
            nsil += h < 0
        else:
            h = x
        if T2 and T2[-1] == t:
            H2[-1] = h
        else:
            T2.append(t)
            H2.append(h)
        cur = h
    return T2, H2, nmove, nsil


def score_all(runs, P, L):
    out = {}
    for ds in DSN:
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        nm = ns = 0
        for song, tabs, T, H in runs[ds]:
            if P is None:
                T2, H2, a, b = T, H, 0, 0
            else:
                T2, H2, a, b = compose(tabs, T, H, P)
            nm += a
            ns += b
            B.score_song(song, T2, H2, L, acc)
            AM.song_counts(song, T2, H2, L, acc2)
        n = acc['notes']
        out[ds] = {'rt_nd': acc['hit'] / acc['d'], 'harsh': acc['harsh'] / acc['d'],
                   'nd_first': acc['hit_first'] / acc['d_first'], 'harsh_first': acc['harsh_first'] / acc['d_first'],
                   'on_time_right': acc['on_time_right'] / n, 'changes': acc['changes'] / n,
                   'late_entry': acc['late_entry'] / n, 'fresh100': 100 * acc2['fresh_clash'] / acc2['notes'],
                   'attacks100': 100 * acc2['attacks'] / acc2['notes'], 'moves100': 100 * nm / n,
                   'sil100': 100 * ns / n}
    head = {k: sum(out[ds][k] for ds in DSN) / 4 for k in out['bach']}
    return head, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid')
    ap.add_argument('out')
    ap.add_argument('--seed2', action='store_true')
    ap.add_argument('--L', type=float, default=0.1)
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
    rec = Rec(r2_mix_v2.build(train), BL.Probe(pri))
    runs = {}
    t1 = time.time()
    for ds, songs in B.bench3.eval_sets('val').items():
        runs[ds] = []
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            T, H, up, uh = B.simulate(rec, meta, notes, a.L)
            runs[ds].append((s, rec.tabs, T, H))
    print(f'reactive pass with probes {time.time() - t1:.1f}s', flush=True)
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
         f"moves {h['moves100']:.1f} sil {h['sil100']:.1f} |")
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
