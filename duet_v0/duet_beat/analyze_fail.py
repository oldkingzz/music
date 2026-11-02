"""Why Duet's notes go wrong (val split only): per-segment logs, failure causes, overlap of the top 3.

  .venv/bin/python duet_beat/analyze_fail.py run policy gru retrieval hmm   # segment logs
  .venv/bin/python duet_beat/analyze_fail.py report                         # summary.json / summary.md
  .venv/bin/python duet_beat/analyze_fail.py all                            # both

A SEGMENT is exactly one term of bench.interval_shares: (song, beat k >= J, one of her TRUE notes)
with the time that note overlaps beat k. Every system is built once and run on bench.heard(song)
(same 10 % mishearing, same per-song seed, same cfg incl. bpm) exactly like bench.score, so the
logged time shares reproduce the leaderboard (checked in the report). Two counterfactual reruns:
  clean   the whole song heard with p = 0 (no mishearing anywhere): bench.heard(song, p=0)
  fix     per beat k that has a not-good segment and a misheard note in its decision window
          (notes with onset < k - D and offset > k - D - 2, i.e. the last 2 beats of audio before the
          decision): only THOSE notes are given back their true pitch, all other noise is kept. The
          song is cut after beat min(K, k + N) (beats <= k are unchanged: every system is causal and
          the HMM planner looks N beats ahead in bar positions only; checked on a sample at run time).
Failure causes (one per not-good segment, first rule that fires): see CAUSES below.
Writes only under bench/analysis/. Never runs the test split.
"""
import argparse
import bisect
import gzip
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
from theory import ALTO, CONSONANT, CT, MAJOR, PC_NAMES  # noqa: E402

OUT = os.path.join(HERE, 'bench', 'analysis')
SYSTEMS = ['policy', 'gru', 'retrieval', 'hmm']
TOP3 = ['policy', 'gru', 'retrieval']
MAIN = ['bach', 'folk', 'pop', 'pop909']             # headline datasets (twinkle logged, not pooled)
NO_DEMO = {'pop'}                                     # Hooktheory: never in examples (no titles/audio)
TOL = 0.05                                            # beats: a note starting within this of the beat
#                                                       onset counts as ON the beat (pop909 jitter)
IVN = ['P1/P8', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7']

CAUSES = [
    ('S', 'silent', 'Duet sings nothing on beat k (h = -1, or no Beat for k).'),
    ('A', 'inside-beat change',
     f'her note starts more than {TOL} beat after beat k\'s onset (a 2nd/3rd note inside the beat, or '
     'her first note after a rest at the onset): Duet holds one note chosen before the beat, so it '
     'could not have heard this note. Sub-split: A_conflict = no alto note F3..F5 is consonant with '
     'every note she sings in beat k (one note per beat cannot serve the beat), A_avoidable = some '
     'single note would have fit all of them.'),
    ('B', 'misheard context',
     'at least one note in the decision window (onset < k - D, offset > k - D - 2) was misheard, AND '
     'the targeted rerun with only those notes restored to their true pitch makes this segment good.'),
    ('C', 'voice crossing',
     'Duet is above her true note and the interval is a perfect fourth (ic 5): an octave lower, on '
     'her other side, it would be a perfect fifth (consonant). (Under the bench metric only the '
     'fourth flips; seconds/sevenths/tritone stay harsh under inversion, 3rds/6ths stay good.)'),
    ('F', 'chromatic her note', 'her true pitch class is not in the major scale of the tonic.'),
    ('D', 'she moved, wrong guess',
     'her note at the beat is a different pitch from the last note she started before the decision '
     'time k - D (she moved after Duet stopped listening), and the system did not anticipate it: '
     'note_guess[k] != her pitch class (policy, gru, retrieval); for hmm (no note_guess): Duet\'s '
     'note is consonant with the previous note, i.e. Duet answered the note she left.'),
    ('D2', 'she moved, choice error',
     'she moved as in D, but the guess was right (or, for hmm, Duet\'s note was not good even for the '
     'previous note): Duet knew/expected the new note and still picked a note that clashes.'),
    ('E1', 'she held, expected a move',
     'her note at the beat is the same pitch as the last note she started before k - D (held or '
     'repeated, so it was known), but note_guess[k] predicted a different pitch class.'),
    ('E2', 'she held, pure choice error',
     'held/repeated as in E1 and the guess was right (hmm: always here): the system knew the note '
     'and still chose a clashing one.'),
    ('G', 'other', 'anything else (no earlier note of hers before the decision).'),
]
CAUSE_KEYS = [c[0] for c in CAUSES]


def cls_of(m, h):
    if h is None or h < 0:
        return 'silent', None
    iv = abs(m - h) % 12
    if iv in CONSONANT:
        return 'good', iv
    return ('fourth' if iv == 5 else 'harsh'), iv


def good(m, h):
    return h is not None and h >= 0 and abs(m - h) % 12 in CONSONANT


def name(m):
    return '-' if m is None or m < 0 else f'{PC_NAMES[m % 12]}{m // 12 - 1}'


def run_once(system, song, c):
    if hasattr(system, 'note_guess'):
        system.note_guess = {}
    beats = system.run(song, c)
    g = dict(getattr(system, 'note_guess', None) or {})
    return {b.k: b for b in beats}, g


def cut_song(song, K2):
    if K2 >= song['K']:
        return song
    return dict(song, K=K2, beats=song['beats'][:K2])


def run_system(sname, check_n=3):
    t0 = time.time()
    mod = importlib.import_module(f'bench_systems.{sname}')
    cfg = bench.config()
    system = mod.build(bench.train_sets())
    has_guess = hasattr(system, 'note_guess')
    print(f'{sname}: built in {time.time() - t0:.1f}s (note_guess: {has_guess})', flush=True)
    lo, hi = ALTO
    Hall = list(range(lo, hi + 1))
    rows, n_fix, n_check, n_check_bad = [], 0, 0, 0
    J = cfg.J
    for ds, songs in bench.eval_sets('val').items():
        t1 = time.time()
        for si, s in enumerate(songs):
            c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
            D = (c.L + c.delta) / (60.0 / c.bpm)
            sec = 60.0 / c.bpm
            hs = bench.heard(s)
            by_k, guess = run_once(system, hs, c)
            by_kc, _ = run_once(system, bench.heard(s, p=0.0), c)
            sop, hsop, t, K = s['sop'], hs['sop'], s['tonic'], s['K']
            ons = [n[0] for n in sop]
            # segments, exactly as bench.interval_shares
            segs = []
            for i, (on, off, m) in enumerate(sop):
                for k in range(max(J, int(on)), int(off) + 1):
                    ov = min(off, k + 1) - max(on, k)
                    if ov > 1e-9:
                        segs.append((k, i, on, off, m, ov))
            per_k = defaultdict(list)
            for sg in segs:
                per_k[sg[0]].append(sg)
            # per beat: decision window, previous note, oracle note
            info = {}
            for k, sl in per_k.items():
                when = k - D
                ip = bisect.bisect_left(ons, when) - 1           # last note with onset < when
                win = [i for i in range(max(0, ip - 40), ip + 1) if sop[i][1] > when - 2 and hsop[i][2] != sop[i][2]]
                sc = {h: sum(ov for (_, _, _, _, m, ov) in sl if good(m, h)) for h in Hall}
                h_or = max(Hall, key=lambda h: (sc[h], -abs(h - 65)))
                tot = sum(sg[5] for sg in sl)
                fit_all = any(all(good(sg[4], h) for sg in sl) for h in Hall)
                b = by_k.get(k)
                hk = b.h if b is not None else -1
                onset_seg = [sg for sg in sl if sg[2] <= k + TOL]
                info[k] = dict(ip=ip, win=win, h_or=h_or, ceil=sc[h_or] / tot, fit_all=fit_all,
                               onset_good=(good(onset_seg[0][4], hk) if onset_seg else None),
                               bad=any(not good(sg[4], hk) for sg in sl))
            # targeted fix reruns
            fixh = {}
            for k, inf in info.items():
                if not (inf['bad'] and inf['win']):
                    continue
                fs = [list(n) for n in hsop]
                for i in inf['win']:
                    fs[i][2] = sop[i][2]
                bk, _ = run_once(system, cut_song(dict(hs, sop=fs), min(K, k + c.N)), c)
                fixh[k] = bk[k].h if k in bk else -1
                n_fix += 1
            # causality / cut check on a sample: cut run without fixes reproduces the full run
            if si < check_n:
                for k in range(J, K, 3):
                    bk, _ = run_once(system, cut_song(hs, min(K, k + c.N)), c)
                    n_check += 1
                    if bk[k].h != by_k[k].h:
                        n_check_bad += 1
            for (k, i, on, off, m, ov) in segs:
                b, bc = by_k.get(k), by_kc.get(k)
                h = b.h if b is not None else -1
                hc = bc.h if bc is not None else -1
                cl, iv = cls_of(m, h)
                inf = info[k]
                ip = inf['ip']
                prev = sop[ip][2] if ip >= 0 else None
                g = guess.get(k)
                hp = by_k.get(k - 1)
                r = dict(ds=ds, sid=s.get('id'), si=si, k=k, pos=s['beats'][k]['pos'] if k < K else None,
                         M=s['M'], bpm=c.bpm, D=round(D, 4), tonic=t,
                         a=round(max(on, k), 4), b=round(min(off, k + 1), 4), dur=round(ov, 4),
                         t_s=round(max(on, k) * sec, 3), dur_s=round(ov * sec, 4),
                         ni=i, n_on=on, n_off=off, her=m, heard=hsop[i][2], mis=hsop[i][2] != m,
                         h=h, ic=iv, cls=cl, above=h >= 0 and h > m,
                         pc=(m - t) % 12, in_key=(m - t) % 12 in MAJOR,
                         z_true=s['beats'][k]['z'] if k < K else -1,
                         h_in_chord=(h >= 0 and k < K and s['beats'][k]['z'] >= 0 and (h - t) % 12 in CT[s['beats'][k]['z']]),
                         n_in_beat=len(per_k[k]), inside=on > k + TOL, onset_unheard=on >= k - D,
                         prev=prev, prev_heard=hsop[ip][2] if ip >= 0 else None,
                         prev_rest=(ip >= 0 and sop[ip][1] <= k - D),
                         same_note=(i == ip), moved=(prev is not None and i != ip and m != prev),
                         guess=g, guess_right=(g == (m - t) % 12) if g is not None else None,
                         guess_stay=(g == (prev - t) % 12) if (g is not None and prev is not None) else None,
                         good_vs_prev=(prev is not None and good(prev, h)),
                         h_prev=hp.h if hp is not None else None, q=round(float(b.q), 4) if b is not None else None,
                         mis_ctx=len(inf['win']), h_fix=fixh.get(k), cls_fix=cls_of(m, fixh[k])[0] if k in fixh else None,
                         h_clean=hc, cls_clean=cls_of(m, hc)[0],
                         h_or=inf['h_or'], or_good=good(m, inf['h_or']), ceil=round(inf['ceil'], 4),
                         fit_all=inf['fit_all'], onset_good=inf['onset_good'])
                r['cause'] = cause_of(r, has_guess)
                rows.append(r)
        print(f'  {sname} {ds}: {len(songs)} songs {time.time() - t1:.1f}s', flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'segments_{sname}.jsonl.gz')
    with gzip.open(path + '.tmp', 'wt') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    os.replace(path + '.tmp', path)
    meta = dict(system=sname, has_guess=has_guess, rows=len(rows), fix_reruns=n_fix,
                cut_check=n_check, cut_check_mismatch=n_check_bad, secs=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(OUT, f'meta_{sname}.json'), 'w'), indent=1)
    print(f'{sname}: {meta}', flush=True)
    return meta


def cause_of(r, has_guess):
    if r['cls'] == 'good':
        return None
    if r['h'] < 0:
        return 'S'
    if r['inside']:
        return 'A'
    if r['mis_ctx'] and r['cls_fix'] == 'good':
        return 'B'
    if r['above'] and r['ic'] == 5:
        return 'C'
    if not r['in_key']:
        return 'F'
    if r['prev'] is None:
        return 'G'
    if r['moved']:
        wrong = (not r['guess_right']) if has_guess else r['good_vs_prev']
        return 'D' if wrong else 'D2'
    if has_guess and not r['guess_right']:
        return 'E1'
    return 'E2'


# ====================================================================== report
def load(sname):
    with gzip.open(os.path.join(OUT, f'segments_{sname}.jsonl.gz'), 'rt') as fh:
        return [json.loads(x) for x in fh]


def pct(x):
    return f'{100 * x:.1f}'


def shares_by(rows, keyf, val=lambda r: r['dur']):
    d = defaultdict(float)
    for r in rows:
        d[keyf(r)] += val(r)
    return d


def board_rows():
    best = {}
    for line in open(bench.BOARD):
        e = json.loads(line)
        if e['split'] == 'val':
            best[e['system']] = e
    return best


def report():
    R = {s: load(s) for s in SYSTEMS if os.path.exists(os.path.join(OUT, f'segments_{s}.jsonl.gz'))}
    meta = {s: json.load(open(os.path.join(OUT, f'meta_{s}.json'))) for s in R}
    board = board_rows()
    S = {'causes': {k: dict(name=n, rule=t) for k, n, t in CAUSES}, 'tol_onset_beats': TOL, 'meta': meta}
    groups = MAIN + ['all']

    def in_g(r, g):
        return r['ds'] == g if g != 'all' else r['ds'] in MAIN

    # ---- 1. reproduce the bench, time shares per class
    cls_tab, check = {}, {}
    for s, rows in R.items():
        cls_tab[s] = {}
        for g in groups + ['twinkle']:
            rs = [r for r in rows if in_g(r, g)]
            tot = sum(r['dur'] for r in rs)
            d = shares_by(rs, lambda r: r['cls'])
            dc = shares_by(rs, lambda r: r['cls_clean'])
            cls_tab[s][g] = {c: d[c] / tot for c in ('good', 'harsh', 'fourth', 'silent')}
            cls_tab[s][g]['good_clean_ear'] = dc['good'] / tot
            cls_tab[s][g]['oracle_one_note'] = sum(r['dur'] * r['or_good'] for r in rs) / tot
            cls_tab[s][g]['time_beats'] = tot
        cls_tab[s]['headline'] = sum(cls_tab[s][g]['good'] for g in MAIN) / len(MAIN)
        cls_tab[s]['headline_clean_ear'] = sum(cls_tab[s][g]['good_clean_ear'] for g in MAIN) / len(MAIN)
        if s in board:
            check[s] = max(abs(cls_tab[s][g][c] - board[s]['rows'][g][c])
                           for g in MAIN + ['twinkle'] for c in ('good', 'harsh', 'fourth', 'silent'))
    S['class_shares'] = cls_tab
    S['bench_reproduction_max_abs_diff'] = check

    # ---- 2. causes of not-good time
    cause_tab, flag_tab = {}, {}
    FLAGS = dict(
        inside=lambda r: r['inside'], misheard_window=lambda r: r['mis_ctx'] > 0,
        fix_window_makes_good=lambda r: r['cls_fix'] == 'good', clean_ear_makes_good=lambda r: r['cls_clean'] == 'good',
        duet_above_her=lambda r: r['above'], her_note_chromatic=lambda r: not r['in_key'],
        her_note_misheard=lambda r: r['mis'], she_moved=lambda r: r['moved'],
        guess_right=lambda r: r['guess_right'] is True, duet_good_vs_prev_note=lambda r: r['good_vs_prev'],
        beat_unservable_by_one_note=lambda r: not r['fit_all'], oracle_note_good_here=lambda r: r['or_good'],
        short_note_lt_quarter_beat=lambda r: (r['n_off'] - r['n_on']) < 0.25)
    for s, rows in R.items():
        cause_tab[s], flag_tab[s] = {}, {}
        for g in groups:
            bad = [r for r in rows if in_g(r, g) and r['cls'] != 'good']
            tb = sum(r['dur'] for r in bad) or float('nan')
            d = shares_by(bad, lambda r: r['cause'])
            cause_tab[s][g] = {c: d[c] / tb for c in CAUSE_KEYS}
            a = [r for r in bad if r['cause'] == 'A']
            ta = sum(r['dur'] for r in a)
            cause_tab[s][g]['A_conflict_share_of_A'] = (sum(r['dur'] for r in a if not r['fit_all']) / ta) if ta else None
            cause_tab[s][g]['A_onset_note_was_good_share_of_A'] = (sum(r['dur'] for r in a if r['onset_good']) / ta) if ta else None
            flag_tab[s][g] = {f: sum(r['dur'] for r in bad if fn(r)) / tb for f, fn in FLAGS.items()}
            # harsh vs fourth inside each cause
            cause_tab[s][g]['fourth_share_by_cause'] = {
                c: (sum(r['dur'] for r in bad if r['cause'] == c and r['cls'] == 'fourth') / d[c]) if d[c] else None
                for c in CAUSE_KEYS}
    S['cause_shares_of_not_good'] = cause_tab
    S['flags_share_of_not_good'] = flag_tab
    S['risk'] = risk_tables(R, groups, in_g)

    # ---- 3. overlap of the top 3
    if all(s in R for s in TOP3):
        ov = overlap(R)
        S['overlap_top3'] = ov
    else:
        ov = None
    S['examples'] = examples(R) if ov else []
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
    write_md(S)
    return S


def risk_tables(R, groups, in_g):
    """Where the risk is: time share of each melodic situation and the system's not-good rate in it."""
    SIT = [('inside_beat', lambda r: r['inside']),
           ('on_beat_held', lambda r: not r['inside'] and not r['moved']),
           ('on_beat_moved', lambda r: not r['inside'] and r['moved'])]
    out = {}
    for s, rows in R.items():
        out[s] = {}
        for g in groups:
            rs = [r for r in rows if in_g(r, g)]
            T = sum(r['dur'] for r in rs)
            x = {}
            for nm, fn in SIT:
                sub = [r for r in rs if fn(r)]
                t = sum(r['dur'] for r in sub)
                x[nm] = dict(time_share=t / T, bad_rate=_ratio(sum(r['dur'] for r in sub if r['cls'] != 'good'), t))
            on = [r for r in rs if not r['inside'] and r['guess'] is not None]
            for nm, fn in (('guess_right', lambda r: r['guess_right']), ('guess_wrong', lambda r: not r['guess_right'])):
                sub = [r for r in on if fn(r)]
                t = sum(r['dur'] for r in sub)
                x[nm] = dict(time_share_of_on_beat=_ratio(t, sum(r['dur'] for r in on)),
                             bad_rate=_ratio(sum(r['dur'] for r in sub if r['cls'] != 'good'), t))
            d = [r for r in rs if r['cause'] == 'D']
            td = sum(r['dur'] for r in d)
            x['D_duet_fit_her_previous_note'] = _ratio(sum(r['dur'] for r in d if r['good_vs_prev']), td)
            x['D_guess_was_stay'] = (_ratio(sum(r['dur'] for r in d if r['guess_stay']), td)
                                     if any(r['guess'] is not None for r in rs) else None)
            e1 = [r for r in rs if r['cause'] == 'E1']
            x['E1_duet_fit_guessed_note'] = _ratio(sum(r['dur'] for r in e1 if good(r['her'] + (r['guess'] - r['pc'] + 6) % 12 - 6, r['h'])),
                                                   sum(r['dur'] for r in e1))
            out[s][g] = x
    # one-note oracle without unison/octave (doubling): per beat, best single alto note
    rows = next(iter(R.values()))
    beats = defaultdict(list)
    for r in rows:
        beats[(r['ds'], r['si'], r['k'])].append((r['her'], r['dur']))
    H = range(ALTO[0], ALTO[1] + 1)
    nd = defaultdict(float)
    tot = defaultdict(float)
    for key, sl in beats.items():
        best = max(sum(d for m, d in sl if abs(m - h) % 12 in CONSONANT - {0}) for h in H)
        for g in groups:
            if key[0] == g or (g == 'all' and key[0] in MAIN):
                nd[g] += best
                tot[g] += sum(d for _, d in sl)
    out['oracle_one_note_no_unison'] = {g: nd[g] / tot[g] for g in groups}
    return out


def overlap(R):
    keyf = lambda r: (r['ds'], r['si'], r['k'], r['ni'])  # noqa: E731
    idx = {s: {keyf(r): r for r in R[s]} for s in TOP3}
    keys = [k for k in idx['policy'] if all(k in idx[s] for s in TOP3)]
    aligned = len(keys) == len(R['policy']) == len(R['gru']) == len(R['retrieval'])
    if 'hmm' in R:
        aligned = aligned and len(R['hmm']) == len(keys) and all(keyf(r) in idx['policy'] for r in R['hmm'])
    out = {'aligned_all_4': aligned, 'segments': len(keys)}
    for g in MAIN + ['all']:
        ks = [k for k in keys if (k[0] == g if g != 'all' else k[0] in MAIN)]
        dur = {k: idx['policy'][k]['dur'] for k in ks}
        bad = {s: {k for k in ks if idx[s][k]['cls'] != 'good'} for s in TOP3}
        T = sum(dur.values())
        tb = lambda ss: sum(dur[k] for k in ss)  # noqa: E731
        nbad = {k: sum(k in bad[s] for s in TOP3) for k in ks}
        U = [k for k in ks if nbad[k] > 0]
        tU = tb(U)
        o = dict(time_beats=T, union_bad_share_of_time=tU / T,
                 all3_share_of_union=tb([k for k in U if nbad[k] == 3]) / tU,
                 exactly2_share_of_union=tb([k for k in U if nbad[k] == 2]) / tU,
                 exactly1_share_of_union=tb([k for k in U if nbad[k] == 1]) / tU)
        o['jaccard'] = {f'{a}-{b}': tb(bad[a] & bad[b]) / tb(bad[a] | bad[b])
                        for i, a in enumerate(TOP3) for b in TOP3[i + 1:]}
        o['per_system'] = {}
        for s in TOP3:
            ts = tb(bad[s])
            others = [x for x in TOP3 if x != s]
            o['per_system'][s] = dict(
                bad_share_of_time=ts / T,
                also_bad_both_others=tb([k for k in bad[s] if nbad[k] == 3]) / ts,
                also_bad_one_other=tb([k for k in bad[s] if nbad[k] == 2]) / ts,
                alone=tb([k for k in bad[s] if nbad[k] == 1]) / ts,
                by_cause={c: dict(share_of_bad=tb([k for k in bad[s] if idx[s][k]['cause'] == c]) / ts,
                                  all3_fail=_ratio(tb([k for k in bad[s] if idx[s][k]['cause'] == c and nbad[k] == 3]),
                                                   tb([k for k in bad[s] if idx[s][k]['cause'] == c])),
                                  alone=_ratio(tb([k for k in bad[s] if idx[s][k]['cause'] == c and nbad[k] == 1]),
                                               tb([k for k in bad[s] if idx[s][k]['cause'] == c])))
                          for c in CAUSE_KEYS},
                others=others)
        A3 = [k for k in U if nbad[k] == 3]
        t3 = tb(A3)
        o['all3'] = dict(
            share_of_time=t3 / T,
            cause_mix={s: {c: tb([k for k in A3 if idx[s][k]['cause'] == c]) / t3 for c in CAUSE_KEYS} for s in TOP3},
            same_cause_label_all3=tb([k for k in A3 if len({idx[s][k]['cause'] for s in TOP3}) == 1]) / t3,
            same_pitch_class_all3=tb([k for k in A3 if len({idx[s][k]['h'] % 12 for s in TOP3}) == 1]) / t3,
            same_midi_all3=tb([k for k in A3 if len({idx[s][k]['h'] for s in TOP3}) == 1]) / t3,
            same_interval_class_all3=tb([k for k in A3 if len({idx[s][k]['ic'] for s in TOP3}) == 1]) / t3,
            oracle_one_note_good=tb([k for k in A3 if idx['policy'][k]['or_good']]) / t3,
            inside_beat=tb([k for k in A3 if idx['policy'][k]['inside']]) / t3,
            chromatic=tb([k for k in A3 if not idx['policy'][k]['in_key']]) / t3,
            she_moved=tb([k for k in A3 if idx['policy'][k]['moved']]) / t3,
            harsh_all3=tb([k for k in A3 if all(idx[s][k]['cls'] == 'harsh' for s in TOP3)]) / t3)
        o['only_one'] = {}
        for s in TOP3:
            ks1 = [k for k in U if nbad[k] == 1 and k in bad[s]]
            t1 = tb(ks1)
            o['only_one'][s] = dict(share_of_union=t1 / tU,
                                    cause_mix={c: _ratio(tb([k for k in ks1 if idx[s][k]['cause'] == c]), t1) for c in CAUSE_KEYS})
        W = [k for k in ks if nbad[k] == 2]                    # exactly one of the three is good
        win = {k: next(s for s in TOP3 if k not in bad[s]) for k in W}
        Won = [k for k in W if not idx['policy'][k]['inside']]
        o['exactly_one_good'] = dict(
            share_of_time=tb(W) / T,
            winner={s: _ratio(tb([k for k in W if win[k] == s]), tb(W)) for s in TOP3},
            winner_guess_right_on_beat=_ratio(tb([k for k in Won if idx[win[k]][k]['guess_right']]), tb(Won)),
            she_moved=_ratio(tb([k for k in W if idx['policy'][k]['moved']]), tb(W)))
        o['exactly_two'] = {f'{a}+{b}': tb([k for k in U if nbad[k] == 2 and k in bad[a] and k in bad[b]]) / tU
                            for i, a in enumerate(TOP3) for b in TOP3[i + 1:]}
        # ensembles
        o['oracle_of_3_good'] = 1.0 - tb(A3) / T
        o['best_single_good'] = max(1.0 - tb(bad[s]) / T for s in TOP3)
        beat_h = {}
        for k in ks:
            bk = k[:3]
            if bk not in beat_h:
                hs = [idx[s][k]['h'] for s in TOP3]
                pcs = [h % 12 for h in hs if h >= 0]
                best = max(set(pcs), key=pcs.count) if pcs else None
                if best is not None and pcs.count(best) >= 2:
                    beat_h[bk] = next(h for h in hs if h >= 0 and h % 12 == best)
                else:
                    beat_h[bk] = hs[0]
        o['majority_vote_good'] = sum(dur[k] for k in ks if good(idx['policy'][k]['her'], beat_h[k[:3]])) / T
        out[g] = o
    out['headline_oracle_of_3'] = sum(out[g]['oracle_of_3_good'] for g in MAIN) / len(MAIN)
    out['headline_majority_vote'] = sum(out[g]['majority_vote_good'] for g in MAIN) / len(MAIN)
    out['headline_pairwise_oracle'] = {}
    for i, a in enumerate(TOP3):
        for b in TOP3[i + 1:]:
            v = []
            for g in MAIN:
                ks = [k for k in keys if k[0] == g]
                T = sum(idx['policy'][k]['dur'] for k in ks)
                v.append(sum(idx['policy'][k]['dur'] for k in ks if idx[a][k]['cls'] == 'good' or idx[b][k]['cls'] == 'good') / T)
            out['headline_pairwise_oracle'][f'{a}|{b}'] = sum(v) / len(v)
    return out


def _ratio(a, b):
    return a / b if b else None


def examples(R, per=3, min_dur=0.5):
    """For each cause (ranked by all-3-fail time), segments where all 3 top systems fail with the SAME
    cause, at least min_dur beats long, from datasets allowed in demos, one per song."""
    keyf = lambda r: (r['ds'], r['si'], r['k'], r['ni'])  # noqa: E731
    idx = {s: {keyf(r): r for r in R[s]} for s in TOP3}
    by_beat = {s: defaultdict(list) for s in TOP3 + (['hmm'] if 'hmm' in R else [])}
    for s in by_beat:
        for r in R[s]:
            by_beat[s][(r['ds'], r['si'])].append(r)
    songs = {(ds, si): s for ds, ss in bench.eval_sets('val').items() for si, s in enumerate(ss)}
    rank = defaultdict(float)
    cand = defaultdict(list)
    for k, r in idx['policy'].items():
        rs = [idx[s][k] for s in TOP3]
        if all(x['cls'] != 'good' for x in rs) and len({x['cause'] for x in rs}) == 1:
            rank[r['cause']] += r['dur']
            if r['ds'] not in NO_DEMO and r['dur'] >= min_dur:
                cand[r['cause']].append(rs)
    out = []
    for c in sorted(rank, key=lambda c: -rank[c]):
        if c in ('S', 'G'):
            continue
        for rs in _pick(cand[c], per):
            out.append(_example(c, rs, songs, by_beat, 'all3'))
    # signature deaths: segments where ONLY one system fails, with its most distinctive cause
    only = {s: defaultdict(list) for s in TOP3}
    tshare = {s: defaultdict(float) for s in TOP3}
    for k, r in idx['policy'].items():
        rs = [idx[s][k] for s in TOP3]
        badl = [s for s, x in zip(TOP3, rs) if x['cls'] != 'good']
        if len(badl) == 1:
            s = badl[0]
            x = idx[s][k]
            tshare[s][x['cause']] += x['dur']
            if x['ds'] not in NO_DEMO and x['dur'] >= min_dur:
                only[s][x['cause']].append([x] + [idx[o][k] for o in TOP3 if o != s])
    for s in TOP3:
        tot = sum(tshare[s].values())
        share = {c: v / tot for c, v in tshare[s].items()}
        oth = {c: sum(tshare[o][c] / sum(tshare[o].values()) for o in TOP3 if o != s) / 2 for c in CAUSE_KEYS}
        cands = [c for c in CAUSE_KEYS if c not in ('S', 'G', 'A') and only[s][c]]
        if not cands:
            continue
        c = max(cands, key=lambda c: share.get(c, 0) - oth[c])
        for rs in _pick(only[s][c], 2):
            e = _example(c, rs, songs, by_beat, f'only {s}')
            e['distinctive'] = dict(share_of_only_this=share.get(c, 0), share_of_only_others_mean=oth[c])
            out.append(e)
    return out


def _pick(cands, per):
    """Longest segments first, one per song, spread over folk / bach / pop909 before repeating a dataset."""
    cs = sorted(cands, key=lambda rs: (-rs[0]['dur'], rs[0]['ds'], rs[0]['si'], rs[0]['k']))
    seen, picked = set(), []
    for ds_pref in ('folk', 'bach', 'pop909', 'twinkle'):
        for rs in cs:
            r = rs[0]
            if r['ds'] == ds_pref and (r['ds'], r['si']) not in seen and len(picked) < per:
                seen.add((r['ds'], r['si']))
                picked.append(rs)
                break
    for rs in cs:
        if len(picked) >= per:
            break
        if (rs[0]['ds'], rs[0]['si']) not in seen:
            seen.add((rs[0]['ds'], rs[0]['si']))
            picked.append(rs)
    return picked


def _example(c, rs, songs, by_beat, kind):
    r = rs[0]
    song = songs[(r['ds'], r['si'])]
    hs = bench.heard(song)['sop']
    k = r['k']
    her = [dict(on=round(on, 3), off=round(off, 3), note=name(m),
                heard=(name(hs[i][2]) if hs[i][2] != m else None))
           for i, (on, off, m) in enumerate(song['sop']) if off > k - 1 and on < k + 2]
    duet = {}
    for s in by_beat:
        bk = {x['k']: x['h'] for x in by_beat[s][(r['ds'], r['si'])]}
        duet[s] = [name(bk.get(j)) for j in (k - 1, k, k + 1)]
    return dict(
        kind=kind, cause=c, ds=r['ds'], song=r['sid'], title=song.get('title') if r['ds'] not in NO_DEMO else None,
        k=k, bpm=r['bpm'], t_s=r['t_s'], seg=[r['a'], r['b']], her_note=name(r['her']),
        her_heard=name(r['heard']) if r['mis'] else None,
        duet_k={s: dict(note=name(x['h']), interval=IVN[x['ic']] if x['ic'] is not None else 'silent',
                        guess=PC_NAMES[(x['guess'] + x['tonic']) % 12] if x['guess'] is not None else None,
                        cause=x['cause'])
                for s, x in zip(_order(kind), rs)},
        her_notes_k_minus1_to_k_plus1=her, duet_notes_k_minus1_k_k_plus1=duet,
        tonic=PC_NAMES[r['tonic']], prev_note=name(r['prev']), oracle_note=name(r['h_or']))


def _order(kind):
    if kind.startswith('only '):
        s = kind[5:]
        return [s] + [o for o in TOP3 if o != s]
    return TOP3


def write_md(S):
    L = ['# Why Duet fails (val, noise 10 %)', '',
         'Segments = (song, beat k, her true note piece overlapping beat k), the terms of bench.interval_shares. '
         'Shares are of her melody TIME. "pooled" = bach+folk+pop+pop909 time pooled; headline = mean of the 4.', '']
    ct = S['class_shares']
    L += ['## 1. What Duet sings against her (share of all scored time)', '',
          '| system | headline good | pooled good | harsh | fourth | silent | good with clean ear | one-note oracle |',
          '|---|---|---|---|---|---|---|---|']
    for s, t in ct.items():
        a = t['all']
        L.append(f"| {s} | {pct(t['headline'])} | {pct(a['good'])} | {pct(a['harsh'])} | {pct(a['fourth'])} | "
                 f"{pct(a['silent'])} | {pct(a['good_clean_ear'])} | {pct(a['oracle_one_note'])} |")
    L += ['', 'Per dataset harsh / fourth / silent (%):', '', '| system | ' + ' | '.join(MAIN) + ' |',
          '|---|' + '---|' * len(MAIN)]
    for s, t in ct.items():
        L.append(f'| {s} | ' + ' | '.join(f"{pct(t[g]['harsh'])} / {pct(t[g]['fourth'])} / {pct(t[g]['silent'])}" for g in MAIN) + ' |')
    onu = S['risk']['oracle_one_note_no_unison']
    L += ['', 'One-note oracle = best single alto note (F3..F5) per beat chosen knowing her whole beat; without '
          'unison/octave: ' + ', '.join(f'{g} {pct(onu[g])}' for g in MAIN + ['all']) + ' %.',
          f"Bench reproduction (max abs diff vs leaderboard, from 4-decimal rounding): "
          + ', '.join(f'{k} {v:.1e}' for k, v in S['bench_reproduction_max_abs_diff'].items()), '']
    rk = S['risk']
    L += ['## 1b. Where the risk is (pooled): time share of each situation and not-good rate in it (%)', '',
          '| system | inside-beat: time / bad | on-beat held: time / bad | on-beat moved: time / bad | '
          'guess right: bad | guess wrong: bad | D: Duet fit her previous note | D: guessed she would stay | '
          'E1: Duet fit the guessed note |', '|---|---|---|---|---|---|---|---|---|']
    for s_ in S['class_shares']:
        x = rk[s_]['all']
        f = lambda v: pct(v) if v is not None else '-'  # noqa: E731
        L.append(f"| {s_} | {f(x['inside_beat']['time_share'])} / {f(x['inside_beat']['bad_rate'])} | "
                 f"{f(x['on_beat_held']['time_share'])} / {f(x['on_beat_held']['bad_rate'])} | "
                 f"{f(x['on_beat_moved']['time_share'])} / {f(x['on_beat_moved']['bad_rate'])} | "
                 f"{f(x['guess_right']['bad_rate'])} | {f(x['guess_wrong']['bad_rate'])} | "
                 f"{f(x['D_duet_fit_her_previous_note'])} | {f(x['D_guess_was_stay'])} | {f(x['E1_duet_fit_guessed_note'])} |")
    L.append('')
    L += ['## 2. Cause rules (first rule that fires)', '']
    for k, n, t in CAUSES:
        L.append(f'- **{k} {n}**: {t}')
    L += ['', '## 3. Causes, share of NOT-GOOD time (%)', '']
    cz = S['cause_shares_of_not_good']
    L += ['| system | dataset | ' + ' | '.join(CAUSE_KEYS) + ' | A conflict / A | A: Duet good on the beat\'s first note |',
          '|---|---|' + '---|' * (len(CAUSE_KEYS) + 2)]
    for s, t in cz.items():
        for g in MAIN + ['all']:
            x = t[g]
            ac, ao = x['A_conflict_share_of_A'], x['A_onset_note_was_good_share_of_A']
            L.append(f'| {s} | {g} | ' + ' | '.join(pct(x[c]) for c in CAUSE_KEYS)
                     + f" | {pct(ac) if ac is not None else '-'} | {pct(ao) if ao is not None else '-'} |")
    L += ['', '## 4. Flags on not-good time (not exclusive, pooled, %)', '']
    fl = S['flags_share_of_not_good']
    names = list(next(iter(fl.values()))['all'].keys())
    L += ['| flag | ' + ' | '.join(fl) + ' |', '|---|' + '---|' * len(fl)]
    for f in names:
        L.append(f'| {f} | ' + ' | '.join(pct(fl[s]['all'][f]) for s in fl) + ' |')
    ov = S.get('overlap_top3')
    if ov:
        a = ov['all']
        L += ['', '## 5. Do the top 3 fail in the same places? (pooled)', '',
              f"Aligned segments across all 4 logs: {ov['aligned_all_4']} ({ov['segments']} segments).", '',
              f"Union of bad time (bad for at least one): {pct(a['union_bad_share_of_time'])} % of time. Of it: "
              f"all 3 bad {pct(a['all3_share_of_union'])} %, exactly 2 {pct(a['exactly2_share_of_union'])} %, "
              f"exactly 1 {pct(a['exactly1_share_of_union'])} %.", '',
              '| pair | Jaccard of bad time |', '|---|---|']
        for p, v in a['jaccard'].items():
            L.append(f'| {p} | {v:.3f} |')
        L += ['', 'Per dataset: union bad % of time; of the union: all 3 / exactly 2 / exactly 1 (%); Jaccard p-g, p-r, g-r:', '',
              '| dataset | union bad | all 3 | exactly 2 | exactly 1 | Jaccard p-g | p-r | g-r |', '|---|---|---|---|---|---|---|---|']
        for g in MAIN + ['all']:
            y = ov[g]
            j = list(y['jaccard'].values())
            L.append(f"| {g} | {pct(y['union_bad_share_of_time'])} | {pct(y['all3_share_of_union'])} | "
                     f"{pct(y['exactly2_share_of_union'])} | {pct(y['exactly1_share_of_union'])} | "
                     f"{j[0]:.2f} | {j[1]:.2f} | {j[2]:.2f} |")
        e1g = a['exactly_one_good']
        L += ['', f"Exactly one of the three is good on {pct(e1g['share_of_time'])} % of time (winner: "
              + ', '.join(f'{s} {pct(v)}' for s, v in e1g['winner'].items())
              + f"); she moved there {pct(e1g['she_moved'])} %; on on-beat segments the winner had guessed her "
              f"note right only {pct(e1g['winner_guess_right_on_beat'])} % of the time (the rest is a lucky hedge).", '']
        L += ['', '| system | bad % of time | also bad for both others | one other | alone |', '|---|---|---|---|---|']
        for s, x in a['per_system'].items():
            L.append(f"| {s} | {pct(x['bad_share_of_time'])} | {pct(x['also_bad_both_others'])} | "
                     f"{pct(x['also_bad_one_other'])} | {pct(x['alone'])} |")
        L += ['', 'Per cause (system\'s own label): share of that cause\'s time where all 3 fail / where only this system fails (%):', '',
              '| system | ' + ' | '.join(CAUSE_KEYS) + ' |', '|---|' + '---|' * len(CAUSE_KEYS)]
        for s, x in a['per_system'].items():
            cells = []
            for c in CAUSE_KEYS:
                y = x['by_cause'][c]
                cells.append(f"{pct(y['all3_fail'])}/{pct(y['alone'])}" if y['all3_fail'] is not None else '-')
            L.append(f'| {s} | ' + ' | '.join(cells) + ' |')
        A3 = a['all3']
        L += ['', f"When all 3 fail ({pct(A3['share_of_time'])} % of time): same cause label {pct(A3['same_cause_label_all3'])} %, "
              f"same pitch class {pct(A3['same_pitch_class_all3'])} %, same MIDI note {pct(A3['same_midi_all3'])} %, "
              f"same interval {pct(A3['same_interval_class_all3'])} %, all harsh {pct(A3['harsh_all3'])} %; "
              f"inside-beat {pct(A3['inside_beat'])} %, she moved {pct(A3['she_moved'])} %, chromatic {pct(A3['chromatic'])} %; "
              f"a one-note oracle would be good on {pct(A3['oracle_one_note_good'])} %.", '',
              '| system | ' + ' | '.join(CAUSE_KEYS) + ' |', '|---|' + '---|' * len(CAUSE_KEYS)]
        for s, m in A3['cause_mix'].items():
            L.append(f'| {s} (all-3-fail mix) | ' + ' | '.join(pct(m[c]) for c in CAUSE_KEYS) + ' |')
        L += ['', 'Only one fails:', '', '| system | share of union | ' + ' | '.join(CAUSE_KEYS) + ' |',
              '|---|---|' + '---|' * len(CAUSE_KEYS)]
        for s, x in a['only_one'].items():
            L.append(f"| {s} | {pct(x['share_of_union'])} | " + ' | '.join(pct(x['cause_mix'][c]) if x['cause_mix'][c] is not None else '-' for c in CAUSE_KEYS) + ' |')
        L += ['', '## 6. Ensemble bounds (good %)', '', '| dataset | best single | oracle of 3 | majority vote |', '|---|---|---|---|']
        for g in MAIN + ['all']:
            L.append(f"| {g} | {pct(ov[g]['best_single_good'])} | {pct(ov[g]['oracle_of_3_good'])} | {pct(ov[g]['majority_vote_good'])} |")
        L += ['', f"Headline (mean of 4): oracle of 3 {pct(ov['headline_oracle_of_3'])}, majority vote "
              f"{pct(ov['headline_majority_vote'])}; pairwise oracles "
              + ', '.join(f'{p} {pct(v)}' for p, v in ov['headline_pairwise_oracle'].items()) + '.',
              'Majority vote: per beat, the pitch class sung by at least 2 of the 3 (the octave of the best-ranked of '
              'them); if all differ, policy. Pick-by-confidence not done: gru/retrieval Beat.q is a chord '
              'probability, not a note confidence, so the three are not comparable without changing their code.']
    ex = S.get('examples') or []
    if ex:
        L += ['', '## 7. Examples to listen for (no Hooktheory songs)', '',
              '[all3] = all three top systems fail with the same cause label; [only X] = only X fails, with X\'s most '
              'distinctive cause among its solo failures (its share minus the mean share of the other two).', '']
        for e in ex:
            her = ' '.join(f"{n['note']}{'(heard ' + n['heard'] + ')' if n['heard'] else ''}@{n['on']}" for n in e['her_notes_k_minus1_to_k_plus1'])
            du = '; '.join(f"{s} {' '.join(v)}" for s, v in e['duet_notes_k_minus1_k_k_plus1'].items())
            dk = ', '.join(f"{s} {v['note']} ({v['interval']}, guess {v['guess']}, {v['cause'] or 'good'})" for s, v in e['duet_k'].items())
            L.append(f"- **{e['cause']}** [{e['kind']}] {e['ds']} `{e['song']}`{(' ' + repr(e['title'])) if e['title'] else ''} beat {e['k']} "
                     f"(t = {e['t_s']} s at {e['bpm']:.0f} bpm, key {e['tonic']}): her {e['her_note']}"
                     f"{' heard as ' + e['her_heard'] if e['her_heard'] else ''} over [{e['seg'][0]}, {e['seg'][1]}); "
                     f"Duet at k: {dk}. Her notes k-1..k+1: {her}. Duet k-1,k,k+1: {du}. One-note oracle: {e['oracle_note']}.")
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'report', 'all'))
    ap.add_argument('systems', nargs='*', default=SYSTEMS)
    a = ap.parse_args()
    if a.cmd in ('run', 'all'):
        for s in a.systems:
            run_system(s)
    if a.cmd in ('report', 'all'):
        report()
        print(open(os.path.join(OUT, 'summary.md')).read())
