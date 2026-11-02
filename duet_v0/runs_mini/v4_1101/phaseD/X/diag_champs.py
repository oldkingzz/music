"""Phase D, agent X: consistent REAL-TIME chord-guess numbers for the cards (v4_1101; reads the bundle, edits nothing).

For each of m1_filter_v5lo, r1_gru_chooser_v1, r2_mix_v2, r3_style_mix_v3: build it as rescore_nd.py does
(mod.build(bench3.train_sets())), wrap THAT INSTANCE's belief() so it records a copy of what it returns (it returns the
same object, so behaviour is unchanged; a step() wrapper only checks that belief() ran exactly once per note), and
replay bench3.run_song itself on every val song of bach, folk, pop, pop909 (heard notes, 10 % mishearing, seed ^ 5160).
Scored with runs_mini/v4_1101/tools/diag_rt.py's definitions (J, GOOD, BUCKETS, feasible_below, feasible_any are
imported from it; its one_song scoring loop is mirrored line for line for ONE belief, the recorded one):
  music_nd, double       = rescore_nd.py (per dataset; headline music_nd = mean of the 4, double = max of the 4)
  miss_share             every missed scored segment in exactly one bucket, in diag_rt's order: onset_beat_unlabelled,
                         held_into_new_chord, no_good_note_below, misheard, wrong_chord_guess (argmax belief != onset
                         chord), right_chord_still_missed; as a share of scored time (they sum to 1 - music_nd)
  chord_acc              notes whose onset beat is >= 4 and labelled: acc = top-1 (argmax of the recorded belief == the
                         chord of the beat her note starts on), p_true = mean belief of that chord, p_top (new here) =
                         mean max of the belief; diag_rt's keys: all, chord_changed / chord_same /
                         first_or_after_unlabelled (vs the onset chord of her previous note), note_1..note_4 (4 = 4th
                         and later) inside its chord span, downbeat / other_beat, misheard / heard_right.
                         Headline = mean of the 4 datasets' values, notes = sum.
Checks: music_nd and double == runs_mini/v4_1101/nd/<name>/results/nd_<name>.json exactly; m1_filter_v5lo == diag_rt_v1
'rt' and r1_gru_chooser_v1's chord accuracy == diag_rt_v1 'gru' (the same 3-GRU belief) exactly; top-1 and top
probability vs round 3's r3/C/diag_20261101_153920.json (4 decimals). Val only (the bundle has no test split);
everything here was tuned on val. Writes runs_mini/v4_1101/diag/<out>.json (mode 'x').

  cd ABS/runs_mini/v4_1101/phaseD/X && ABS/.venv/bin/python ABS/runs_mini/v4_1101/tools/hbrun.py pD_X_diag 600 \
      ABS/.venv/bin/python ABS/runs_mini/v4_1101/phaseD/X/diag_champs.py diag_champs_v1
"""
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
_BAD = sorted(k for k in os.environ if k.startswith(('M1_', 'M3_')) or k == 'DUET_HIST')
if _BAD:
    sys.exit(f'refusing to run: environment has {_BAD}')
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
sys.path.append(os.path.join(RUN, 'tools'))
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
import diag_rt  # noqa: E402  (definitions only; its main() does not run)
import torch  # noqa: E402
from theory import CT  # noqa: E402

SYSTEMS = ('m1_filter_v5lo', 'r1_gru_chooser_v1', 'r2_mix_v2', 'r3_style_mix_v3')
DS = ('bach', 'folk', 'pop', 'pop909')
J, GOOD, BUCKETS = diag_rt.J, diag_rt.GOOD, diag_rt.BUCKETS
SYS_DIR = os.path.join(ABS, 'futures', 'v3_nolatency', 'systems')
DEPS = ('m1_filter_v5lo', 'r1_gru_chooser_v1', 'r2_mix_v2', 'r3_style_mix_v3', 'r2_style_hmm_v1', 'm3_seq_v4', 'm3_seq_v1')
SHA_STATE = {'r1_gru_chooser_v1': '2d26a11f22fe', 'r2_mix_v2': 'b481d9156f3c', 'r3_style_mix_v3': 'b79ea9bf01c4',
             'r2_style_hmm_v1': 'a37e0f1b3168', 'diag_rt': '2c68b0d46c9d'}          # as recorded in state_R1..R4
R3C = os.path.join(RUN, 'r3', 'C', 'diag_20261101_153920.json')
R3C_LABEL = {'m1_filter_v5lo': 'pooled_hmm m1_filter_v5lo', 'r1_gru_chooser_v1': 'gru3 r1_gru_chooser_v1',
             'r2_mix_v2': 'champion r2_mix_v2 (pooled, a 0.7)', 'r3_style_mix_v3': 'r3_style_mix_v3 (style, a 0.5)'}
COLS = (('acc', 'all'), ('chg', 'chord_changed'), ('same', 'chord_same'), ('n1', 'note_1_in_chord'),
        ('n2', 'note_2_in_chord'), ('n3', 'note_3_in_chord'), ('n4+', 'note_4_in_chord'))


def sha12(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


class Recorder:
    """Wraps ONE system instance. belief(k) records a copy of what it returns and returns the very same object;
    step() checks that belief() ran exactly once inside it and files that belief under the note. Nothing else changes."""

    def __init__(self, system):
        self.bels, self.calls, self.last, self.dtype = [], 0, None, None
        belief, step = system.belief, system.step        # bound methods of the instance's own class

        def rec_belief(k):
            b = belief(k)
            self.calls += 1
            self.last = np.array(b, copy=True)
            self.dtype = self.dtype or str(self.last.dtype)
            return b

        def rec_step(i, onset, midi, prev_dur):
            c0 = self.calls
            out = step(i, onset, midi, prev_dur)
            if self.calls != c0 + 1:
                raise RuntimeError(f'belief() ran {self.calls - c0} times in step {i}')
            self.bels.append(self.last)
            return out

        system.belief = rec_belief                         # instance attributes: self.belief(k) inside step() finds it
        system.step = rec_step


def new_acc():
    return {'nd': [0.0, 0.0, 0.0], 'miss': {b: 0.0 for b in BUCKETS}, 'no_good_below_but_above_ok': 0.0,
            'acc': {}, 'notes': 0}                         # acc: key -> [notes, right, sum P(true), sum max P]


def score_song(song, outs, bels, r):
    """diag_rt.one_song's scoring loop, line for line, for one belief source (the recorded one)."""
    h = bench3.B2.heard(song)
    heard = sorted(h['sop'], key=lambda n: n[0])
    true = sorted(song['sop'], key=lambda n: n[0])
    t = song['tonic']
    z = [b['z'] for b in song['beats']]
    pos = [b['pos'] for b in song['beats']]
    span, sid = [], -1                    # chord span of each beat: a run of beats with the same label
    for k in range(len(z)):
        if z[k] < 0:
            span.append(None)
            continue
        if k == 0 or z[k - 1] != z[k]:
            sid += 1
        span.append(sid)
    seen, kprev = {}, None
    for i, (on, off, m) in enumerate(true):
        mh = heard[i][2]
        k0 = int(on)
        z0 = z[k0] if k0 < len(z) else -1
        b = bels[i]
        if z0 >= 0:
            seen[span[k0]] = seen.get(span[k0], 0) + 1
            if on >= J:
                r['notes'] += 1
                zp = z[kprev] if kprev is not None and kprev < len(z) else -1
                ch = 'first_or_after_unlabelled' if zp < 0 else ('chord_changed' if zp != z0 else 'chord_same')
                keys = ('all', ch, 'downbeat' if pos[k0] == 0 else 'other_beat',
                        f'note_{min(seen[span[k0]], 4)}_in_chord', 'misheard' if mh != m else 'heard_right')
                ok = int(int(np.argmax(b)) == z0)
                for key in keys:
                    a = r['acc'].setdefault(key, [0, 0, 0.0, 0.0])
                    a[0] += 1
                    a[1] += ok
                    a[2] += float(b[z0])
                    a[3] += float(np.max(b))
        kprev = k0
        for k in range(max(J, k0), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            hh = outs[i]
            a = r['nd']
            a[1] += ov
            good = False
            if hh >= 0:
                iv = abs(m - hh) % 12
                good = iv in GOOD and (hh - t) % 12 in CT[z[k]]
                a[0] += ov * good
                a[2] += ov * (iv == 0)
            if good:
                continue
            if z0 < 0:
                bk = 'onset_beat_unlabelled'
            elif z[k] != z0:
                bk = 'held_into_new_chord'
            elif not diag_rt.feasible_below(m, z[k], t):
                bk = 'no_good_note_below'
            elif mh != m:
                bk = 'misheard'
            elif int(np.argmax(b)) != z0:
                bk = 'wrong_chord_guess'
            else:
                bk = 'right_chord_still_missed'
            r['miss'][bk] += ov
            if bk == 'no_good_note_below' and diag_rt.feasible_any(m, z[k], t):
                r['no_good_below_but_above_ok'] += ov


def finish(r):
    num, den, dbl = r['nd']
    out = {'notes_scored': r['notes'], 'music_nd': num / den, 'double': dbl / den,
           'miss_share': {b: v / den for b, v in r['miss'].items()},
           'no_good_below_but_above_ok_share': r['no_good_below_but_above_ok'] / den,
           'chord_acc': {k: {'notes': a[0], 'acc': a[1] / a[0], 'p_true': a[2] / a[0], 'p_top': a[3] / a[0]}
                         for k, a in sorted(r['acc'].items())}}
    out['buckets_plus_nd_minus_1'] = sum(out['miss_share'].values()) + out['music_nd'] - 1.0
    return out


def mean_over(res, f):
    return sum(f(v) for v in res.values()) / len(res)


def headline(res):
    return {'music_nd': mean_over(res, lambda v: v['music_nd']),
            'double': max(v['double'] for v in res.values()),
            'miss_share': {b: mean_over(res, lambda v: v['miss_share'][b]) for b in BUCKETS},
            'no_good_below_but_above_ok_share': mean_over(res, lambda v: v['no_good_below_but_above_ok_share']),
            'chord_acc': {k: {'acc': mean_over(res, lambda v: v['chord_acc'][k]['acc']),
                              'p_true': mean_over(res, lambda v: v['chord_acc'][k]['p_true']),
                              'p_top': mean_over(res, lambda v: v['chord_acc'][k]['p_top']),
                              'notes': sum(v['chord_acc'][k]['notes'] for v in res.values())}
                          for k in res['bach']['chord_acc']}}


def nd_check(name, res, head):
    """[recorded, replayed] music_nd and double vs the system's own rescore_nd.py json; exact float equality."""
    with open(os.path.join(RUN, 'nd', name, 'results', f'nd_{name}.json')) as fh:
        rec = json.load(fh)
    chk = {ds: {'music_nd': [rec[ds]['music_nd'], res[ds]['music_nd']], 'double': [rec[ds]['double'], res[ds]['double']],
                'equal': rec[ds]['music_nd'] == res[ds]['music_nd'] and rec[ds]['double'] == res[ds]['double']}
           for ds in DS}
    chk['headline_nd'] = {'music_nd': [rec['headline_nd'], head['music_nd']], 'equal': rec['headline_nd'] == head['music_nd']}
    chk['all_equal'] = all(v['equal'] for v in chk.values())
    return chk


def summarize(pairs):
    """pairs: (label, mine, reference, tolerance) -> how many agree exactly, the worst gap, all within tolerance?"""
    d = [(abs(a - b), lab, tol) for lab, a, b, tol in pairs]
    worst = max(d)
    return {'compared': len(d), 'exactly_equal': sum(x == 0 for x, _, _ in d), 'max_abs_diff': worst[0],
            'worst': worst[1], 'all_within_tolerance': all(x <= tol for x, _, tol in d)}


def cross_diag_rt(systems):
    """m1_filter_v5lo == diag_rt_v1 'rt' (music_nd, double, buckets, chord accuracy); r1_gru_chooser_v1's chord accuracy
    == diag_rt_v1 'gru' (diag_rt ran the same 3-GRU average on the same heard notes). Tolerance 0 (exact)."""
    with open(os.path.join(RUN, 'diag', 'diag_rt_v1.json')) as fh:
        ref = json.load(fh)
    pairs = []
    for lvl in DS + ('headline',):
        th = ref['headline_mean_of_4'] if lvl == 'headline' else ref['datasets'][lvl]
        mine = {n: (v['headline_mean_of_4'] if lvl == 'headline' else v['datasets'][lvl]) for n, v in systems.items()}
        m = mine['m1_filter_v5lo']
        pairs += [(f'm1 {lvl} music_nd', m['music_nd'], th['music_nd']['rt'], 0.0),
                  (f'm1 {lvl} double', m['double'], th['double']['rt'], 0.0),
                  (f'm1 {lvl} no_good_below_but_above_ok', m['no_good_below_but_above_ok_share'],
                   th['no_good_below_but_above_ok_share'], 0.0)]
        pairs += [(f'm1 {lvl} miss {b}', m['miss_share'][b], th['miss_share']['rt'][b], 0.0) for b in BUCKETS]
        for g, n in (('rt', 'm1_filter_v5lo'), ('gru', 'r1_gru_chooser_v1')):
            for key, v in th['chord_acc'][g].items():
                w = mine[n]['chord_acc'][key]
                pairs += [(f'{n} {lvl} {key} {f}', w[f], v[f], 0.0) for f in ('notes', 'acc', 'p_true')]
    return summarize(pairs)


def cross_r3c(systems):
    """Round 3's r3/C/diag.py (same notes; it normalised the belief and rounded to 4 decimals; its mean4 is a mean of
    rounded values): top-1 accuracy, mean top probability and note counts."""
    with open(R3C) as fh:
        ref = json.load(fh)
    pairs = []
    for name, lab in R3C_LABEL.items():
        v = systems[name]
        for ds in DS:
            ca = v['datasets'][ds]['chord_acc']['all']
            pairs += [(f'{name} {ds} notes', ca['notes'], ref[lab][ds]['notes'], 0.0),
                      (f'{name} {ds} top1_acc', ca['acc'], ref[lab][ds]['top1_acc'], 5.01e-5),
                      (f'{name} {ds} mean_top_prob', ca['p_top'], ref[lab][ds]['mean_top_prob'], 5.01e-5)]
        hv = v['headline_mean_of_4']['chord_acc']['all']
        pairs += [(f'{name} mean4 top1_acc', hv['acc'], ref[lab]['mean4']['top1_acc'], 1.01e-4),
                  (f'{name} mean4 mean_top_prob', hv['p_top'], ref[lab]['mean4']['mean_top_prob'], 1.01e-4)]
    return summarize(pairs)


def row(v):
    ca = v['chord_acc']
    d = {'music_nd': round(v['music_nd'], 4), 'double': round(v['double'], 4)}
    d.update({c: round(ca[k]['acc'], 4) for c, k in COLS})
    d.update({'p_true': round(ca['all']['p_true'], 4), 'p_top': round(ca['all']['p_top'], 4)})
    d.update({b: round(x, 4) for b, x in v['miss_share'].items()})
    return d


def table(systems):
    return {n: dict({'headline': row(v['headline_mean_of_4'])}, **{ds: row(v['datasets'][ds]) for ds in DS})
            for n, v in systems.items()}


def show(tab):
    ab = ('unlab', 'held', 'nobelow', 'mishrd', 'wrong', 'rightms')
    print(f"{'system':18s} {'level':8s} {'nd':>6s} {'dbl':>6s} " + ' '.join(f'{c:>6s}' for c, _ in COLS)
          + f" {'Ptrue':>6s} {'Ptop':>6s} | miss " + ' '.join(f'{x:>7s}' for x in ab))
    for n, rows in tab.items():
        for lvl, d in rows.items():
            print(f"{n:18s} {lvl:8s} {d['music_nd']:6.4f} {d['double']:6.4f} "
                  + ' '.join(f'{d[c]:6.4f}' for c, _ in COLS) + f" {d['p_true']:6.4f} {d['p_top']:6.4f} |      "
                  + ' '.join(f'{d[b]:7.4f}' for b in BUCKETS))


def main():
    name = sys.argv[1]
    path = os.path.join(RUN, 'diag', name + '.json')
    if os.path.exists(path):
        sys.exit(f'refusing: {path} exists')
    t0 = time.time()
    torch.set_num_threads(1)                   # 1 thread for GRU inference (PROGRAM 1.5); the GRU systems' build() too
    shas = {n: sha12(os.path.join(SYS_DIR, n + '.py')) for n in DEPS}
    shas['diag_rt'] = sha12(diag_rt.__file__)
    shas['diag_champs'] = sha12(os.path.abspath(__file__))
    evals = bench3.eval_sets('val')
    if tuple(evals) != DS:
        sys.exit(f'unexpected val datasets {tuple(evals)}')
    systems = {}
    for sname in SYSTEMS:
        t1 = time.time()
        system = importlib.import_module(f'systems.{sname}').build(bench3.train_sets())
        rec = Recorder(system)
        res, fed = {}, 0
        for ds, songs in evals.items():
            r = new_acc()
            for song in songs:
                rec.bels = []
                hs = bench3.run_song(system, song)
                if len(rec.bels) != len(hs):
                    raise RuntimeError(f'{sname}: {len(rec.bels)} beliefs for {len(hs)} notes')
                fed += len(hs)
                score_song(song, hs, rec.bels, r)
            res[ds] = finish(r)
        head = headline(res)
        systems[sname] = {'class': type(system).__name__, 'secs': round(time.time() - t1, 1), 'notes_fed': fed,
                          'belief_calls': rec.calls, 'one_belief_per_note': rec.calls == fed,
                          'belief_dtype': rec.dtype, 'nd_check': nd_check(sname, res, head),
                          'headline_mean_of_4': head, 'datasets': res}
        print(sname, 'nd reproduced:', systems[sname]['nd_check']['all_equal'], round(head['music_nd'], 6),
              'top-1', round(head['chord_acc']['all']['acc'], 4), f"{systems[sname]['secs']}s", flush=True)
    out = {'what': __doc__.strip().split('\n')[0],
           'notes': 'val only (the bundle has no test split); everything was tuned on val. Headline = mean over bach, '
                    'folk, pop, pop909 (double: max over the 4, as diag_rt.py; chord_acc notes: sum). Definitions: '
                    'runs_mini/v4_1101/tools/diag_rt.py; p_top (mean max belief) is new here.',
           'reproduces_all_nd': all(v['nd_check']['all_equal'] for v in systems.values()),
           'one_belief_per_note_all': all(v['one_belief_per_note'] for v in systems.values()),
           'crosscheck_diag_rt_v1': cross_diag_rt(systems),
           'crosscheck_r3C_diag': cross_r3c(systems),
           'sha256_12': shas, 'sha_matches_state': {n: shas[n] == v for n, v in SHA_STATE.items()},
           'table_4dp': table(systems), 'systems': systems, 'secs': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    show(out['table_4dp'])
    for k in ('reproduces_all_nd', 'one_belief_per_note_all', 'crosscheck_diag_rt_v1', 'crosscheck_r3C_diag',
              'sha_matches_state', 'secs'):
        print(k, json.dumps(out[k]))
    print('wrote', path)


if __name__ == '__main__':
    main()
