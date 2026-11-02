"""Round 3, idea F DIAGNOSTIC v3 (v2 + settings + the real-voice file), not a scored bench. Val only, tuned on val.
Like diag_live_F_v2.py (read it) but the system can be rt_systems.r3_F_lib_v1.PlanF with JSON overrides on
r2_copyhm_v1's CFG (tune_F_v1.BASE), and --voice phone_v11 lists every attack on the real-voice file with the decision
that made it and its interval against her HEARD note (live_rt_eval_v2.heard_proxy's rules: her notes = the segmenter's
closed notes, fresh_heard = an attack harsh against the heard pitch of the note it falls in).
Offline only: the _NoSound tripwire is installed before any other import; nothing is played.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/diag_live_F_v3.py --cfg '<json>' [--cfg ...] --voice phone_v11 --out <new json>
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
import os  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
HERE = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'r3', 'F')
sys.path.insert(0, HERE)
import diag_live_F_v2 as D2  # noqa: E402  (its own tripwire is identical; installs live_bench_rt)
import tune_F_v1 as T1  # noqa: E402

LBR, EV = D2.LBR, D2.EV
OUT_DIR = HERE


def phone_v11(rec, label):
    cfg = EV.V11
    x = EV.read(cfg['file']).astype('float64')
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    eng = LBR.make_engine(rec, label, facts, 0)
    track, notes, blips, secs = LBR.run_engine(eng, x, LBR.PHONE_LAT)
    prox = EV.heard_proxy(eng, track, notes, blips, EV.LO)
    plan_calls = [c for c in rec.calls if c[0] == 'plan']
    hear_calls = [c for c in rec.calls if c[0] == 'hear']
    dec = eng.decisions
    pi = hi = 0
    for d in dec:
        if d['kind'] == 'plan':
            d['path'] = plan_calls[pi][3] if pi < len(plan_calls) else []
            pi += 1
        elif d['kind'] == 'hear':
            d['path'] = hear_calls[hi][3] if hi < len(hear_calls) else []
            hi += 1
        else:
            d['path'] = []
    dts = [d['t'] for d in dec]
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    rows = []
    for xs, h in EV.attacks(sorted(notes + blips)):
        j = bisect.bisect_left(dts, xs - 0.0006)
        da = None
        while j < len(dec) and dts[j] <= xs + 0.0006:
            if dec[j]['action'] in ('enter', 'switch', 'reenter'):
                da = dec[j]
            j += 1
        nt = EV.her_at(xs, her)
        iv = None if nt is None else abs(nt[2] - h) % 12
        rows.append({'t': round(xs, 3), 'beat': round((xs - eng.t0) / eng.T_b, 3), 'h': int(h),
                     'her_heard': None if nt is None else nt[2], 'iv': iv, 'harsh': iv in D2.BR.HARSH,
                     'above': None if nt is None else bool(h > nt[2]),
                     'origin': 'none' if da is None else da['kind'] + ('_reenter' if da['action'] == 'reenter' else ''),
                     'q': None if da is None else da['q'], 'path': [] if da is None else da['path']})
    return prox, rows, [[e['onset_beats'], e['her_heard'], e['duet'], e['action']] for e in eng.events]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', action='append', default=[])
    ap.add_argument('--spec', action='append', default=[])
    ap.add_argument('--voice', required=True, choices=D2.VV.VOICES + ('phone_v11',))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_DIR + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new .json under {OUT_DIR}/')
    systems = [(s, LBR.load_system(s)) for s in a.spec]
    if a.cfg:
        train = LBR.bench3.train_sets()
        mix = T1.r2_mix_v2.build(train)
        prior, tab, hm = T1.Prior(), T1.load_copy_tab(), T1.LIB3.load_hm_tab()
        for js in a.cfg:
            systems.append(('cfg:' + js, T1.FL.PlanF(mix, prior, dict(T1.BASE, **json.loads(js)), tab, hm)))
    res = {'voice': a.voice, 'note': 'val only, tuned on val; diagnostic', 'systems': {}}
    for label, sysm in systems:
        rec = D2.Rec(sysm)
        if a.voice == 'phone_v11':
            prox, rows, ans = phone_v11(rec, label)
            res['systems'][label] = {'proxies': prox, 'attacks': rows, 'answered': ans}
            print(label, 'v11 fresh_heard', prox['fresh_heard'], 'harsh_heard %.4f' % prox['harsh_heard'],
                  'attacks', prox['attacks'], flush=True)
            for r in rows:
                if r['harsh']:
                    print('   harsh attack', r, flush=True)
            print('   answered', ans, flush=True)
        else:
            lat = LBR.LAT['wired']
            acc = LBR.new_acc()
            rows = []
            for s in LBR.bench3.eval_sets('val')['bach']:
                D2.one_song(rec, label, s, a.voice, lat, acc, rows)
            m = LBR.metrics(acc)
            sm = D2.summarize(rows, m['notes'])
            res['systems'][label] = {'metrics': m, 'summary': sm, 'rows': rows}
            print(label, a.voice, 'rt_nd_live %.4f harsh %.4f fresh/100 %.2f stumbles/100 %.2f wrong/100 %.1f' % (
                m['rt_nd_live'], m['harsh'], m['fresh_per100'], m['stumbles_per100'], m['wrong_att_per100']),
                flush=True)
            for k in ('st_by', 'st_pc', 'st_path', 'st_by_path', 'fresh', 'fresh_path', 'fresh_err', 'plan_ctx', 'above'):
                print('  ' + k, dict(list(sm[k].items())[:14]), flush=True)
            print('  attacks/100', sm['per100']['attacks'], '| wrong/100', sm['per100']['wrong'], flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LBR._js)


if __name__ == '__main__':
    main()
