"""diag_v11_X_v1 (run v5rt_1101, round 3, combine step): where the real-voice proxies (fresh_heard, harsh_heard) of a
combined variant come from on the phone file v11 (no ground truth), note by note, with the base system's branch at
every hear (stats delta, the tracker posterior's P(misheard), the register shift). Uses live_bench_rt's own phone_v11
set-up (make_engine, run_engine, PHONE_LAT) and live_rt_eval_v2.heard_proxy / attacks / her_at, like
runs_mini/v5rt_1101/r3/D/diag_v11_v1.py. Offline: _NoSound tripwire before any other import; nothing is played.
Specs as dev_X_v1.py. Tuned on val (no test split); v11 is one real recording. Writes one new JSON.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/r3/X/diag_v11_X_v1.py --out <new json> '<spec>' ['<spec>' ...]
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('diag_v11_X_v1 is offline: no sound device may be used (sounddevice.' + name + ')')


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

from systems import r2_mix_v2  # noqa: E402
from rt_systems import r2_lib_v3 as R3  # noqa: E402
from rt_systems import r3_D_lib_v3 as DL  # noqa: E402
from rt_systems import r3_E_lib_v2 as EL  # noqa: E402
from rt_systems import r3_X_lib_v1 as XL  # noqa: E402
from rt_systems.r2_lib_v1 import Prior, load_copy_tab  # noqa: E402
from rt_systems.r3_D_reghm_v1 import CFG as REG0  # noqa: E402

EV = LBR.EV
HARSH = LBR.HARSH
NAMES = ('C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B')


def nn(m):
    return f'{NAMES[m % 12]}{m // 12 - 1}' if m is not None and m >= 0 else '-'


def instrument(base, log):
    hear0, plan0 = base.hear, base.plan

    def hear(i, onset, midi, prev_dur):
        st0 = dict(base.stats)
        r = hear0(i, onset, midi, prev_dur)
        d = {k: v - st0.get(k, 0) for k, v in base.stats.items() if v != st0.get(k, 0)}
        lp = getattr(base, 'last_post', None)
        log.append({'kind': 'hear', 'i': i, 'onset': round(float(onset), 4), 'midi_base': int(midi), 'ret': r,
                    'h': int(base.h), 'branch': d,
                    'p_mis': None if lp is None else round(float(1.0 - lp[EL.E0]), 3)})
        return r

    def plan(q, heard):
        st0 = dict(base.stats)
        r = plan0(q, heard)
        if r is not None:
            d = {k: v - st0.get(k, 0) for k, v in base.stats.items() if v != st0.get(k, 0)}
            log.append({'kind': 'plan', 'q': round(float(q), 4), 'ret': r, 'h': int(base.h), 'branch': d})
        return r

    base.hear, base.plan = hear, plan


def one(system, base, name):
    cfg = EV.V11
    x = EV.read(cfg['file']).astype(np.float64)
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    log = []
    instrument(base, log)
    eng = LBR.make_engine(system, name, facts, 0)
    track, notes, blips, secs = LBR.run_engine(eng, x, LBR.PHONE_LAT)
    prox = EV.heard_proxy(eng, track, notes, blips, EV.LO)
    t_out = (np.arange(len(track)) + 2) * LBR.HOP / LBR.SR + EV.LO
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    ans = {round(c0, 4): (eng.events[i] if i < len(eng.events) else None) for c0, i in eng.by_start.items()}
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
            'adapter': getattr(system, 'songs', None), 'base_log': log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('specs', nargs='+')
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if os.path.exists(path) or not path.endswith('.json'):
        sys.exit('refusing: ' + path)
    train = LBR.bench3.train_sets()
    mix = r2_mix_v2.build(train)
    prior, tab, hm, trk = Prior(), load_copy_tab(), R3.load_hm_tab(), EL.load_trk_tab()
    out = {'note': 'v11 real voice, no truth: proxies vs the heard notes; systems tuned on val (no test split)', 'runs': []}
    for js in a.specs:
        spec = json.loads(js)
        cfg = dict(importlib.import_module('rt_systems.r3_E_' + spec['e'] + '_v1').CFG, **spec.get('over', {}))
        base = XL.make_base(mix, prior, cfg, tab, hm, trk)
        system = base if spec.get('reg') is None else DL.RegAdapter(base, dict(REG0, **spec['reg']))
        res = one(system, base, spec.get('tag', 'x'))
        res['spec'] = spec
        out['runs'].append(res)
        p = res['proxies']
        print(f"=== {spec.get('tag')}: fresh_heard {p['fresh_heard']} harsh_heard {p['harsh_heard']:.4f} attacks "
              f"{p['attacks']} sung {p['sung']} cons_heard {p['cons_heard']} adapter {res['adapter']}")
        for n in res['notes']:
            flag = ' HARSH' if n['harsh_samples'] else ''
            print(f"  k{n['k']:2d} {n['on']:6.2f}-{n['off']:6.2f} her {nn(n['m']):4s} answered {nn(n['answered']):4s} "
                  f"beat {n['beat']} duet {[nn(h) for h in n['duet']]} {n['action']} harsh {n['harsh_samples']}/"
                  f"{n['samples']}{flag}")
        for t in res['attacks']:
            if t['fresh']:
                print(f"  FRESH at {t['t']:.3f}: Duet {nn(t['duet'])} her {nn(t['her'])} iv {t['iv']} from {t['src']}")
        for e in res['base_log']:
            if e['kind'] == 'hear':
                print(f"    hear i={e['i']} on={e['onset']} base_midi {nn(e['midi_base'])} ret {nn(e['ret']) if e['ret'] is not None else 'keep'}"
                      f" p_mis {e['p_mis']} {e['branch']}")
            else:
                print(f"    plan q={e['q']} ret {nn(e['ret'])} {e['branch']}")
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=LBR._js)
    print('wrote', path)


if __name__ == '__main__':
    main()
