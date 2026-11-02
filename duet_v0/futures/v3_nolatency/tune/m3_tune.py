"""Tune m3_seq voice-leading weights on val: cache chord probs once, replay the decision with many cfgs."""
import itertools, os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'duet_beat'))
import bench3, bench as B2
import torch
from systems import m3_seq_v1 as S
from systems import m3_seq_v4 as E
torch.set_num_threads(2)
net = S.get_net(None, os.environ.get('M3_TAG', 'full'))
ev = bench3.eval_sets('val')
cache = {}
for ds, songs in ev.items():
    cache[ds] = []
    for s in songs:
        h = B2.heard(s); notes = sorted(h['sop'], key=lambda n: n[0])
        d = (E.Ens([E._load('full', 96), E._load('big', 128)], -1, S.CFG) if os.environ.get('M3_ENS') else S.Duet(net, -1, S.CFG)); d.start({'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]})
        ps, prev = [], None
        for on, off, m in notes:
            ps.append(d.chord_probs(on, m, None if prev is None else prev[1] - prev[0])); prev = (on, off)
        cache[ds].append((s, notes, ps))

class Replay(S.Duet):
    def chord_probs(self, onset, midi, prev_dur):
        return self.P[self.j]

def run(cfg):
    rows = {}
    for ds, items in cache.items():
        tot = {}
        for s, notes, ps in items:
            d = Replay(None, -1, cfg); d.start({'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]})
            d.P = ps; hs = []
            for j, (on, off, m) in enumerate(notes):
                d.j = j; hs.append(d.step(j, on, m, None))
            for k, v in bench3.song_stats(s, hs).items(): tot[k] = tot.get(k, 0) + v
        rows[ds] = dict(music=tot['music'] / tot['music_d'], big=tot['big'] / max(tot['moves'], 1),
                        par=tot['par'] / max(tot['pairs'], 1), double=tot['double'] / tot['d'])
    return sum(r['music'] for r in rows.values()) / len(rows), rows

res = []
G = json.loads(os.environ['GRID']) if os.environ.get('GRID') else [[0.2, 0.35, 0.6], [0.0, 0.01, 0.02, 0.04], [0.1, 0.3], [0.2, 0.4], [0.0, 0.05, 0.15]]
for dbl, step, big, par, above in itertools.product(*G):
    cfg = dict(dbl=dbl, step=step, big=big, par=par, above=above)
    hl, rows = run(cfg)
    ok = all(r['big'] <= 0.10 and r['par'] <= 0.05 and r['double'] <= 0.25 for r in rows.values())
    res.append((hl, ok, cfg, {k: round(v['music'], 3) for k, v in rows.items()}, max(r['big'] for r in rows.values()), max(r['par'] for r in rows.values())))
    print(f'{hl:.4f} {ok} {cfg} dbl {max(r["double"] for r in rows.values()):.3f} big {res[-1][4]:.3f} par {res[-1][5]:.3f}', flush=True)
res.sort(key=lambda r: (not r[1], -r[0]))
print('BEST', res[:5])
