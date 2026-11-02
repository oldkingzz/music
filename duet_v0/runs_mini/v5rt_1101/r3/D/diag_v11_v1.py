"""Idea D diagnostic on the real-voice phone file v11 (no ground truth): for each engine run, where live_rt_eval_v2's
proxies (fresh_heard, harsh_heard) come from, note by note. Uses live_bench_rt's own engine set-up for phone_v11
(make_engine, run_engine, PHONE_LAT) and live_rt_eval_v2.heard_proxy / attacks / her_at. Offline: _NoSound tripwire
before any other import; nothing is played. Read-only except its one new JSON.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/D/diag_v11_v1.py --runs '<json list as dev_live_v1>' --out <new>
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('diag_v11 is offline: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import live_bench_rt as LBR  # noqa: E402
from rt_systems import r3_D_lib_v1 as DL  # noqa: E402

EV = LBR.EV
HARSH = LBR.HARSH
NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')


def nn(m):
    return f'{NAMES[m % 12]}{m // 12 - 1}' if m is not None and m >= 0 else '-'


def one(system, name):
    cfg = EV.V11
    x = EV.read(cfg['file']).astype(np.float64)
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    eng = LBR.make_engine(system, name, facts, 0)
    track, notes, blips, secs = LBR.run_engine(eng, x, LBR.PHONE_LAT)
    prox = EV.heard_proxy(eng, track, notes, blips, EV.LO)
    t_out = (np.arange(len(track)) + 2) * LBR.HOP / LBR.SR + EV.LO
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    ans = {}
    for c0, i in eng.by_start.items():
        ans[round(c0, 4)] = eng.events[i] if i < len(eng.events) else None
    dec = {round(d['t'], 4): d for d in eng.decisions}
    rows = []
    for k, (on, off, m, _) in enumerate(her):
        idx = np.where((t_out >= on) & (t_out < off))[0]
        hs = [int(track[i]) for i in idx]
        bad = sum(1 for h in hs if h >= 0 and abs(m - h) % 12 in HARSH)
        ev = None
        for c0, e in ans.items():
            if abs(c0 - on) < 0.015:
                ev = e
        rows.append({'k': k, 'on': round(on, 3), 'off': round(off, 3), 'm': m,
                     'answered': None if ev is None else ev['her_heard'], 'beat': None if ev is None else ev['onset_beats'],
                     'duet': sorted(set(hs)), 'harsh_samples': bad, 'samples': len(hs),
                     'action': None if ev is None else ev['action']})
    att = []
    for xa, h in EV.attacks(notes):
        nt = EV.her_at(xa, her)
        d = dec.get(round(xa, 4))
        att.append({'t': round(xa, 3), 'duet': h, 'her': None if nt is None else nt[2],
                    'iv': None if nt is None else abs(nt[2] - h) % 12,
                    'fresh': bool(nt is not None and abs(nt[2] - h) % 12 in HARSH),
                    'src': None if d is None else f"{d['kind']} q={d['q']} i={d['i']} {d['action']}"})
    return {'proxies': prox, 'notes': rows, 'attacks': att, 'decision_counts': eng.summary()['decision_counts'],
            'adapter': getattr(system, 'songs', None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    train = LBR.bench3.train_sets()
    cache, out = {}, {'note': 'v11 real voice, no truth: proxies vs the heard notes; tuned on val', 'runs': []}
    for r in json.loads(a.runs):
        b = r['base']
        if b not in cache:
            cache[b] = LBR.BR.load(b, train) if b.startswith('reactive:') else \
                importlib.import_module('rt_systems.' + b).build(train)
        system = cache[b] if r.get('cfg') is None else DL.RegAdapter(cache[b], r['cfg'])
        res = one(system, r['name'])
        res['name'] = r['name']
        out['runs'].append(res)
        p = res['proxies']
        print(f"=== {r['name']}: fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f} attacks {p['attacks']} "
              f"sung {p['sung']} cons_heard {p['cons_heard']} adapter {res['adapter']}")
        for n in res['notes']:
            flag = ' HARSH' if n['harsh_samples'] else ''
            print(f"  k{n['k']:2d} {n['on']:6.2f}-{n['off']:6.2f} her {nn(n['m']):4s} answered {nn(n['answered']):4s} "
                  f"beat {n['beat']} duet {[nn(h) for h in n['duet']]} {n['action']} harsh {n['harsh_samples']}/"
                  f"{n['samples']}{flag}")
        for t in res['attacks']:
            if t['fresh']:
                print(f"  FRESH at {t['t']:.3f}: Duet {nn(t['duet'])} her {nn(t['her'])} iv {t['iv']} from {t['src']}")
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
