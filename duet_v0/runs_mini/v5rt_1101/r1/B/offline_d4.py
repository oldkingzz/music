"""Idea B offline evaluator v4 (round 1, run v5rt_1101; val only, tuned on val) = offline_d3 with r2_mix_v2's
silence (h_r = -1) handled in the hear rule. One reactive r2_mix_v2 pass per song with probes (blib_d2) at every plan
and hear point; plan and hear rules are replayed exactly by merging their decisions into the reactive step function and
scoring with bench_rt.score_song and attack_metric.song_counts. The filter's prev stays h_r in this replay.

  python offline_d4.py <grid json> <new out json> [--seed2] [--L 0.1] [--calib]
"""
import argparse
import bisect
import importlib.util
import json
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
import blib_d2 as BL  # noqa: E402
from systems import m1_filter_v5lo as M1  # noqa: E402
from systems import r2_mix_v2  # noqa: E402

DSN = ('bach', 'folk', 'pop', 'pop909')
GOOD = BL.GOOD
HARSH = BL.HARSH


class Rec:
    def __init__(self, mix, probe):
        self.f, self.probe = mix, probe

    def start(self, meta):
        self.f.start(meta)
        self.probe.start(meta)
        self.t = meta['tonic']
        self.tabs, self.hears, self.mine = [], [], []

    def plan(self, q, heard):
        self.tabs.append((q, self.probe.table(self.f, q, heard)))
        return None

    def hear(self, i, onset, midi, prev_dur):
        if self.mine and prev_dur is not None:
            on0, _, m0 = self.mine[-1]
            self.mine[-1] = (on0, on0 + prev_dur, m0)
        ht = self.probe.hear_table(self.f, onset, midi, self.mine)
        h = int(self.f.step(i, onset, midi, prev_dur))
        pct = self.f.belief(int(onset)) @ M1.CTM
        self.mine.append((onset, None, midi))
        self.hears.append((i, h, midi, pct, ht))
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


def hear_rule(info, cur, t, P, cnt):
    """Output at a hear point: an int (Duet's note from now, -1 = silence) or None = keep the sounding note."""
    i, h_r, midi, pct, ht = info
    if ht is not None and P.get('SUSP') is not None and ht[0] < P['SUSP']:
        pc, eh, ex = ht
        val = eh - P.get('LAMH', 1.0) * ex
        att = val - P.get('MUH', 0.0) * ex                        # a new attack also risks a fresh clash
        if h_r < 0:
            v_r = -P.get('KSH', 0.0)
        else:
            v_r = att[h_r - BL.G0] if h_r != cur else val[h_r - BL.G0]
        alts = []
        if cur >= 0:
            alts.append((val[cur - BL.G0], None))
        if P.get('SILH', 1):
            alts.append((-P.get('KSH', 0.0), -1 if cur >= 0 else None))
        if P.get('ROBH', 1):
            a2 = att.copy()
            if cur >= 0:
                a2[cur - BL.G0] = -1e9
            j = int(np.argmax(a2))
            alts.append((a2[j], int(BL.G[j])))
        bv, bo = max(alts, key=lambda x: x[0])
        if bv > v_r + P.get('MARGH', 0.0):
            cnt['susp_alt'] += 1
            return bo
    if cur == h_r:
        return None
    if P.get('EPSK') is not None and cur >= 0 and h_r >= 0:
        iv = abs(midi - cur) % 12
        if iv in GOOD and cur < midi and pct[(cur - t) % 12] >= pct[(h_r - t) % 12] - P['EPSK']:
            cnt['keep'] += 1
            return None
    return h_r


def compose(song, tabs, hears, T, P, cal, cnt):
    true = sorted(song['sop'], key=lambda n: n[0])
    ons = [n[0] for n in true]
    t = song['tonic']
    ev = [(q, 0, tab) for q, tab in tabs if tab is not None] + [(T[j], 1, hears[j]) for j in range(len(T))]
    ev.sort(key=lambda e: (e[0], e[1]))
    T2, H2, cur = [], [], -1
    for tt, kind, x in ev:
        if kind == 0:
            if P.get('RHO', -1) < 0 and not P.get('SIL', 0):
                continue
            h, r = plan_rule(x, cur, P)
            if h is None:
                continue
            if h >= 0:
                cnt['move'] += 1
                a = bisect.bisect_right(ons, tt + 1e-9) - 1
                if a >= 0 and true[a][1] > tt and tt >= B.J:
                    cal['risk'] += r
                    cal['n'] += 1
                    cal['harsh'] += abs(true[a][2] - h) % 12 in HARSH
            else:
                cnt['sil'] += 1
        else:
            h = hear_rule(x, cur, t, P, cnt)
            if h is None:
                continue
        if T2 and T2[-1] == tt:
            H2[-1] = h
        else:
            T2.append(tt)
            H2.append(h)
        cur = h
    return T2, H2


def score_all(runs, P, L):
    out = {}
    cal = {'risk': 0.0, 'n': 0, 'harsh': 0}
    for ds in DSN:
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        acc2 = dict.fromkeys(('notes', 'attacks', 'fresh_clash', 'attack_miss', 'd', 'harsh_fresh', 'harsh_bleed'), 0.0)
        cnt = dict.fromkeys(('move', 'sil', 'keep', 'susp_alt'), 0)
        for song, tabs, hears, T, H in runs[ds]:
            T2, H2 = (T, H) if P is None else compose(song, tabs, hears, T, P, cal, cnt)
            B.score_song(song, T2, H2, L, acc)
            AM.song_counts(song, T2, H2, L, acc2)
        n = acc['notes']
        out[ds] = {'rt_nd': acc['hit'] / acc['d'], 'harsh': acc['harsh'] / acc['d'],
                   'nd_first': acc['hit_first'] / acc['d_first'], 'harsh_first': acc['harsh_first'] / acc['d_first'],
                   'on_time_right': acc['on_time_right'] / n, 'changes': acc['changes'] / n,
                   'late_entry': acc['late_entry'] / n, 'fresh100': 100 * acc2['fresh_clash'] / acc2['notes'],
                   'attacks100': 100 * acc2['attacks'] / acc2['notes'], 'moves100': 100 * cnt['move'] / n,
                   'sil100': 100 * cnt['sil'] / n, 'keep100': 100 * cnt['keep'] / n,
                   'alt100': 100 * cnt['susp_alt'] / n}
    head = {k: sum(out[ds][k] for ds in DSN) / 4 for k in out['bach']}
    head['cal_moves'] = cal['n']
    head['cal_risk'] = cal['risk'] / max(cal['n'], 1)
    head['cal_harsh'] = cal['harsh'] / max(cal['n'], 1)
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
            assert len(T) == len(rec.hears), (len(T), len(rec.hears))
            runs[ds].append((s, rec.tabs, rec.hears, T, H))
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
         f"att {h['attacks100']:.1f} moves {h['moves100']:.1f} sil {h['sil100']:.1f} keep {h['keep100']:.1f} "
         f"alt {h['alt100']:.1f} cal {h.get('cal_risk', 0):.3f}/{h.get('cal_harsh', 0):.3f} |")
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
