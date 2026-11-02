"""One-off faithfulness check + exploration, v4 round 4, implementer B (E5 r4_true_cons); not a system, not a scored variant; val songs,
tuned on val (the bundle has no other held-out songs).
(1) Every r4_true_cons_v<k> file present, with both parts off (instance attributes belief_side = choice_side = 'off'; classes and
    modules untouched), must sing exactly r3_style_mix_v3's notes through bench3.run_song on the first 40 val songs of each dataset
    (bach has 17), and its 3-GRU ensemble must advance exactly once per note (also checked with each file's own settings).
(2) Exploration on ALL val songs (rescore_nd.py's music_nd and double, same loops): the champion copy (both off, = reproduction of
    nd_r3_style_mix_v3.json) and a few belief/choice settings set as instance attributes on one built object.
Writes runs_mini/v4_1101/r4/B/check_copy_<time>.json (new file, mode 'x'); exits 1 if (1) fails anywhere.
"""
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3DIR = os.path.join(ABS, 'futures', 'v3_nolatency')
OUTDIR = os.path.join(ABS, 'runs_mini', 'v4_1101', 'r4', 'B')
sys.path.insert(0, V3DIR)
import bench3  # noqa: E402
from theory import CONSONANT, CT, MAJOR  # noqa: E402
from systems import r2_style_hmm_v1, r3_style_mix_v3  # noqa: E402
from systems import r4_true_cons_v1  # noqa: E402

MODS = {'r4_true_cons_v1': r4_true_cons_v1}
for _k in (2, 3, 4):
    try:
        MODS[f'r4_true_cons_v{_k}'] = __import__('systems.r4_true_cons_v%d' % _k, fromlist=['build'])
    except ImportError:
        pass

N_FIRST = 40
EXPLORE = sys.argv[1:] != ['--copy-only']


class GruSideOnly(r4_true_cons_v1.TrueConsBelief):
    """Exploration only: the GRU part switches to the last trusted guess at a suspect note, the HMM keeps the note's evidence."""

    def _beat_v(self, k):
        return r2_style_hmm_v1.StyleFilter._beat_v(self, k)


CONFIGS = [  # name, class, belief_side, gru_suspect, choice_side
    ('champion_copy', None, 'off', 'prev', 'off'),
    ('cur_prev', None, 'cur', 'prev', 'off'),            # = r4_true_cons_v1
    ('cur_prev_v4', None, 'cur', 'prev', 'v4'),          # = r4_true_cons_v2
    ('cur_heard', None, 'cur', 'heard', 'off'),          # HMM side only
    ('gru_side_only', GruSideOnly, 'cur', 'prev', 'off'),
    ('all_prev', None, 'all', 'prev', 'off'),            # suspect notes never enter the HMM
    ('all_prev_v4', None, 'all', 'prev', 'v4'),
    ('choice_v4_only', None, 'off', 'prev', 'v4'),       # r3_true_cons_v4's choice on the new champion
]


def counter(system):
    calls = [0]
    orig = system.ens.chord_probs

    def counted(onset, midi, prev_dur):
        calls[0] += 1
        return orig(onset, midi, prev_dur)
    system.ens.chord_probs = counted          # instance attribute on this one Ens object
    return calls


def nd_rows(system, sets):
    """rescore_nd.py's numbers for one system object, plus how many notes were suspect."""
    out = {}
    for ds, songs in sets.items():
        num = den = dbl = 0.0
        notes = sus = 0
        for s in songs:
            hs = bench3.run_song(system, s)
            notes += len(hs)
            sus += sum(getattr(system, '_sus', []))
            true = sorted(s['sop'], key=lambda n: n[0])
            z = [b['z'] for b in s['beats']]
            t = s['tonic']
            for i, (on, off, m) in enumerate(true):
                h = hs[i]
                for k in range(max(bench3.J, int(on)), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                        continue
                    den += ov
                    if h >= 0:
                        iv = abs(m - h) % 12
                        num += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                        dbl += ov * (iv == 0)
        out[ds] = {'music_nd': round(num / den, 6), 'double': round(dbl / den, 5), 'suspect_share': round(sus / notes, 4)}
    out['headline_nd'] = round(sum(v['music_nd'] for v in out.values()) / 4, 6)
    return out


def main():
    t0 = time.time()
    train = bench3.train_sets()
    sets = bench3.eval_sets('val')
    first = {ds: songs[:N_FIRST] for ds, songs in sets.items()}
    champ = r3_style_mix_v3.build(train)
    ref = {ds: [bench3.run_song(champ, s) for s in songs] for ds, songs in first.items()}
    res = {'check': 'r4 B: r4_true_cons_v<k> with both parts off vs r3_style_mix_v3 (first 40 val songs per dataset; val, tuned on val)',
           'songs': {d: len(v) for d, v in first.items()}, 'copy': {}, 'explore': {}}
    ok = True
    for name, mod in MODS.items():
        system = mod.build(train)
        calls = counter(system)
        own = (system.belief_side, system.gru_suspect, system.choice_side)
        r = {'file_settings': own, 'off': {}, 'own_gru_calls_ok': {}}
        system.belief_side, system.choice_side = 'off', 'off'
        for ds, songs in first.items():
            c0 = calls[0]
            outs = [bench3.run_song(system, s) for s in songs]
            notes = sum(len(o) for o in outs)
            diff = sum(sum(a != b for a, b in zip(o, q)) + abs(len(o) - len(q)) for o, q in zip(outs, ref[ds]))
            r['off'][ds] = {'notes': notes, 'notes_differing_vs_champion': diff, 'gru_calls': calls[0] - c0}
            ok = ok and diff == 0 and calls[0] - c0 == notes and notes > 0
        del system.belief_side, system.choice_side          # back to the file's own settings (class attributes)
        for ds, songs in first.items():
            c0 = calls[0]
            outs = [bench3.run_song(system, s) for s in songs]
            notes = sum(len(o) for o in outs)
            changed = sum(sum(a != b for a, b in zip(o, q)) for o, q in zip(outs, ref[ds]))
            r['own_gru_calls_ok'][ds] = {'notes': notes, 'gru_calls': calls[0] - c0, 'changed_vs_champion': changed}
            ok = ok and calls[0] - c0 == notes
        res['copy'][name] = r
        print(name, 'both off identical to r3_style_mix_v3:', all(v['notes_differing_vs_champion'] == 0 for v in r['off'].values()),
              '| own settings', own, 'changed', {d: v['changed_vs_champion'] for d, v in r['own_gru_calls_ok'].items()}, flush=True)
    res['copy_ok'] = ok
    if EXPLORE:
        base = r4_true_cons_v1.build(train)
        for cname, cls, bside, gside, cside in CONFIGS:
            system = base if cls is None else cls(base.models, base.lams, base.rh, base.mprior, base.ens)
            system.belief_side, system.gru_suspect, system.choice_side = bside, gside, cside
            t1 = time.time()
            res['explore'][cname] = nd_rows(system, sets)
            res['explore'][cname]['secs'] = round(time.time() - t1, 1)
            e = res['explore'][cname]
            print(cname, e['headline_nd'], {d: e[d]['music_nd'] for d in sets}, 'double', {d: e[d]['double'] for d in sets},
                  'suspect', {d: e[d]['suspect_share'] for d in sets}, e['secs'], 's', flush=True)
    res['secs'] = round(time.time() - t0, 1)
    path = os.path.join(OUTDIR, f"check_copy_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print('COPY_OK', ok, 'secs', res['secs'], 'wrote', path)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
