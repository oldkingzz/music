"""Result analysis, audio demos: where do the top systems sound harsh, and do they fail at the same places?

1. Runs policy, gru, retrieval and hmm on every val song of twinkle, folk (Nottingham) and pop909, each
   on the HEARD copy (bench.heard: 10 % mishearing, no chord labels) with the bench config, exactly as
   bench.score does. Segments and shares follow bench.interval_shares (scored against her TRUE melody
   from beat J on); the totals per dataset are checked against bench.interval_shares.
2. Picks one folk tune (16-40 s at the bench bpm) and one POP909 song (cropped to its first POP_CROP
   beats) where the three top systems differ most: the share of her melody time where their classes
   (good / fourth / harsh / silent) are not all the same. Twinkle is always the third song.
3. Renders, per song: her true melody alone, and her melody + Duet's notes + a soft click per system
   (her clean synthetic voice as audio.render_her, Duet's voice as audio.mix, stereo, 24 kHz, 96 kbps).
4. Writes bench/analysis/demo/timeline.json: notes, heard notes with misheard flags, per-beat Duet notes,
   per-segment classes, shares, how often the top three are harsh at the same moment (per demo song and
   over every val song of the three datasets), one cause per harsh segment (CAUSE_DOC: no single note
   fits the beat / her note began after the decision time / it was misheard / heard right and still
   harsh), and the share of Duet's beats outside the song's major or minor scale.

Never touches the test split. Writes only bench/analysis/demo/ (+ wav scratch in --wav-dir).
  ../.venv/bin/python analyze_demos.py --wav-dir /path/to/scratch
"""
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
from audio import click, duet_notes, mix, render_her  # noqa: E402
from dsp import SR  # noqa: E402
from theory import CONSONANT  # noqa: E402

SYSTEMS = ['policy', 'gru', 'retrieval', 'hmm']
TOP = SYSTEMS[:3]
OUT = os.path.join(HERE, 'bench', 'analysis', 'demo')
POP_CROP = 40                      # beats: 8 listen-in beats (J) + 32 beats of Duet
SEC_LO, SEC_HI = 16.0, 40.0
DUET_LEVEL, CLICK_GAIN = 0.45, 0.5
CLASSES = ('good', 'fourth', 'harsh', 'silent')
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def nname(m):
    return '-' if m is None or m < 0 else f'{NAMES[m % 12]}{m // 12 - 1}'


def cls_of(iv):
    if iv in CONSONANT:
        return 'good'
    if iv in bench.HARSH:
        return 'harsh'
    return 'fourth'                                   # iv == 5: the only class left


def segments(song, beats, J):
    """The loop of bench.interval_shares, keeping every (note, beat) piece and its class."""
    by_k = {b.k: b for b in beats}
    out = []
    for i, (on, off, m) in enumerate(song['sop']):
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9:
                continue
            b = by_k.get(k)
            if b is None or b.h < 0:
                out.append((i, k, max(on, k), min(off, k + 1), 'silent', None))
            else:
                iv = abs(m - b.h) % 12
                out.append((i, k, max(on, k), min(off, k + 1), cls_of(iv), iv))
    return out


CAUSES = ('no_fit', 'unheard', 'unheard_noisy_ctx', 'misheard', 'heard_ok')
CAUSE_DOC = {
    'no_fit': 'her notes inside beat k are such that no single alto note is consonant with all of them '
              '(one note per beat cannot be good everywhere)',
    'unheard': 'a fitting note existed, but this note of hers began after the decision time k - D, so Duet '
               'had to guess it, and none of the notes it heard in the 2 beats before deciding was misheard',
    'unheard_noisy_ctx': 'as unheard, but at least one note it heard in the 2 beats before deciding was misheard',
    'misheard': 'the note had begun before the decision time but Duet heard it as a wrong pitch class',
    'heard_ok': 'the note had begun before the decision time and was heard correctly; Duet still chose a '
                'harsh note against it',
}


def harsh_causes(song, heard_sop, segs, D, alto):
    """One cause per harsh segment, tested in the order of CAUSES (first that applies)."""
    lo, hi = alto
    sop = song['sop']
    fit = {}
    out = []
    for i, k, st, en, c, iv in segs:
        if c != 'harsh':
            out.append(None)
            continue
        if k not in fit:
            ms = [m for on, off, m in sop if min(off, k + 1) - max(on, k) > 1e-9]
            fit[k] = any(all(abs(m - h) % 12 in CONSONANT for m in ms) for h in range(lo, hi + 1))
        if not fit[k]:
            out.append('no_fit')
        elif sop[i][0] >= k - D - 1e-9:
            noisy = any(k - D - 2 <= on < k - D and hm % 12 != tm % 12
                        for (on, _, tm), (_, _, hm) in zip(sop, heard_sop))
            out.append('unheard_noisy_ctx' if noisy else 'unheard')
        elif heard_sop[i][2] % 12 != sop[i][2] % 12:
            out.append('misheard')
        else:
            out.append('heard_ok')
    return out


def cause_shares(segs, causes):
    d = sum(e - s for _, _, s, e, _, _ in segs) or float('nan')
    return {c: sum(e - s for (_, _, s, e, _, _), cc in zip(segs, causes) if cc == c) / d for c in CAUSES}


def shares(segs):
    d = sum(e - s for _, _, s, e, _, _ in segs) or float('nan')
    r = {c: sum(e - s for _, _, s, e, cc, _ in segs if cc == c) / d for c in CLASSES}
    r['double'] = sum(e - s for _, _, s, e, _, iv in segs if iv == 0) / d
    r['dur'] = d if d == d else 0.0
    return r


def crop(song, Kc):
    if song['K'] <= Kc:
        return song, None
    sop = [[on, min(off, Kc), m] for on, off, m in song['sop'] if on < Kc]
    return dict(song, sop=sop, beats=song['beats'][:Kc], K=Kc), len(sop)


def overlap(segs_by_sys, names=TOP):
    """Do the top systems fail at the same moments? Time shares over the scored melody."""
    n = len(segs_by_sys[names[0]])
    d = tot = 0.0
    acc = dict(differ=0.0, any_harsh=0.0, all_harsh=0.0, one_harsh=0.0, any_bad=0.0, all_bad=0.0)
    for j in range(n):
        s, e = segs_by_sys[names[0]][j][2:4]
        w = e - s
        cs = [segs_by_sys[x][j][4] for x in names]
        d += w
        acc['differ'] += w * (len(set(cs)) > 1)
        nh = sum(c == 'harsh' for c in cs)
        nb = sum(c != 'good' for c in cs)
        acc['any_harsh'] += w * (nh >= 1)
        acc['all_harsh'] += w * (nh == len(names))
        acc['one_harsh'] += w * (nh == 1)
        acc['any_bad'] += w * (nb >= 1)
        acc['all_bad'] += w * (nb == len(names))
    tot = d or float('nan')
    r = {k: v / tot for k, v in acc.items()}
    r['same_harsh_given_any'] = acc['all_harsh'] / acc['any_harsh'] if acc['any_harsh'] else None
    r['only_one_given_any'] = acc['one_harsh'] / acc['any_harsh'] if acc['any_harsh'] else None
    r['same_bad_given_any'] = acc['all_bad'] / acc['any_bad'] if acc['any_bad'] else None
    return r


MAJOR, MINOR = {0, 2, 4, 5, 7, 9, 11}, {0, 2, 3, 5, 7, 8, 10}


def song_scale(song):
    """Major or natural minor on the tonic, whichever covers more of her melody time."""
    t = song['tonic']
    cover = {nm: sum(off - on for on, off, m in song['sop'] if (m - t) % 12 in sc) for nm, sc in
             (('major', MAJOR), ('minor', MINOR))}
    nm = max(cover, key=cover.get)
    return nm, (MAJOR if nm == 'major' else MINOR)


def out_of_scale(song, beats, J):
    """Share of Duet's sung beats (k >= J) on a pitch class outside the song's scale."""
    _, sc = song_scale(song)
    sung = [b for b in beats if b.k >= J and b.k < song['K'] and b.h >= 0]
    return sum((b.h - song['tonic']) % 12 not in sc for b in sung) / len(sung) if sung else None


def bar_labels(song):
    """'bar.beat' per beat (bars counted from the first downbeat as 1; a pickup is bar 0)."""
    out, bar = [], 0
    for b in song['beats']:
        if b['pos'] == 0:
            bar += 1
        out.append(f"{bar}.{b['pos'] + 1}")
    return out


def slugify(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def run_all(systems, songs, cfg):
    res = []
    for s in songs:
        c = replace(cfg, bpm=s['bpm']) if 'bpm' in s else cfg
        h = bench.heard(s)
        res.append({name: sysobj.run(h, c) for name, sysobj in systems.items()})
    return res


def ffmpeg_mp3(wav, mp3):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-ar', str(SR), '-ac', '2', '-c:a', 'libmp3lame',
                    '-b:a', '96k', mp3], check=True)


def duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def render(song, beats_by_sys, cfg, slug, wav_dir):
    bpm = song.get('bpm', cfg.bpm)
    T_b = 60.0 / bpm
    count_in = song['M']
    t0 = count_in * T_b
    her = render_her(song, t0, T_b, 'clean')
    n = max(len(her), int((t0 + song['K'] * T_b + 0.8) * SR))
    her = np.pad(her, (0, n - len(her)))
    clk = CLICK_GAIN * click(n, t0, T_b, song['K'], song['M'], count_in)
    files = {}
    for name, beats in [('her', [])] + list(beats_by_sys.items()):
        wav = os.path.join(wav_dir, f'{slug}__{name}.wav')
        mp3 = os.path.join(OUT, f'{slug}__{name}.mp3')
        mix(her, duet_notes([b for b in beats if b.k < song['K']], t0, T_b), clk, wav, None, duet_level=DUET_LEVEL)
        ffmpeg_mp3(wav, mp3)
        files[name] = os.path.basename(mp3)
    return files, t0, T_b, count_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', required=True)
    ap.add_argument('--folk', help='force a folk id instead of the rule')
    ap.add_argument('--pop909', help='force a pop909 id instead of the rule')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.wav_dir, exist_ok=True)
    cfg = bench.config()
    train = bench.train_sets()
    systems = {n: importlib.import_module(f'bench_systems.{n}').build(train) for n in SYSTEMS}
    ev = bench.eval_sets('val')

    # ---- 1. every val song of the three demo datasets, whole songs (as the bench scores them)
    table, check, agg, agg_causes = {}, {}, {}, {}
    for ds in ('twinkle', 'folk', 'pop909'):
        runs = run_all(systems, ev[ds], cfg)
        table[ds] = list(zip(ev[ds], runs))
        tot = {n: {c: 0.0 for c in CLASSES + ('d',)} for n in SYSTEMS}
        segs_all = {n: [] for n in SYSTEMS}
        caus_all = {n: [] for n in SYSTEMS}
        for s, r in table[ds]:
            hs = bench.heard(s)['sop']
            D = (cfg.L + cfg.delta) / (60.0 / s.get('bpm', cfg.bpm))
            for n in SYSTEMS:
                sg = segments(s, r[n], cfg.J)
                ref = bench.interval_shares(s, r[n], cfg.J)
                mine = shares(sg)
                for c in CLASSES:
                    assert abs(mine[c] * mine['dur'] - ref[c]) < 1e-6, (s['id'], n, c)
                    tot[n][c] += ref[c]
                tot[n]['d'] += ref['d']
                segs_all[n] += sg
                caus_all[n] += harsh_causes(s, hs, sg, D, cfg.alto)
        check[ds] = {n: {c: tot[n][c] / tot[n]['d'] for c in CLASSES} for n in SYSTEMS}
        agg[ds] = overlap(segs_all)
        p = [check[ds][n]['harsh'] for n in TOP]
        agg[ds]['all_harsh_if_independent'] = float(np.prod(p))
        agg[ds]['any_harsh_if_independent'] = float(1 - np.prod([1 - x for x in p]))
        agg_causes[ds] = {n: cause_shares(segs_all[n], caus_all[n]) for n in SYSTEMS}
        print(ds, {n: round(check[ds][n]['good'], 3) for n in SYSTEMS},
              'same-harsh|any %.3f  only-one|any %.3f' % (agg[ds]['same_harsh_given_any'], agg[ds]['only_one_given_any']),
              flush=True)
        for n in SYSTEMS:
            print('   harsh by cause', n, {c: round(v, 3) for c, v in agg_causes[ds][n].items()})

    # ---- 2. pick the songs
    def candidates(ds, Kc=None):
        rows = []
        for s, r in table[ds]:
            sc, _ = crop(s, Kc) if Kc else (s, None)
            secs = sc['K'] * 60.0 / sc.get('bpm', cfg.bpm)
            if not (SEC_LO <= secs <= SEC_HI):
                continue
            sg = {n: segments(sc, r[n], cfg.J) for n in SYSTEMS}
            ov = overlap(sg)
            g = {n: shares(sg[n])['good'] for n in TOP}
            rows.append(dict(id=s['id'], title=s.get('title'), K=sc['K'], secs=secs, differ=ov['differ'],
                             spread=max(g.values()) - min(g.values()), good=g, any_harsh=ov['any_harsh'],
                             same=ov['same_harsh_given_any']))
        rows.sort(key=lambda x: (-x['differ'], -x['spread'], x['id']))
        return rows

    pick = {}
    for ds, Kc, forced in (('folk', None, a.folk), ('pop909', POP_CROP, a.pop909)):
        rows = candidates(ds, Kc)
        print(f'\n{ds}: {len(rows)} candidates, top 8 by "classes differ" share')
        for x in rows[:8]:
            print(f"  {x['id']:34s} {str(x['title'])[:24]:24s} K {x['K']:3d} {x['secs']:5.1f}s differ {x['differ']:.3f} "
                  f"spread {x['spread']:.3f} good " + ' '.join(f'{n} {v:.2f}' for n, v in x['good'].items()) +
                  f"  any-harsh {x['any_harsh']:.3f} same|any {x['same'] if x['same'] is not None else float('nan'):.2f}")
        pick[ds] = forced or rows[0]['id']
        pick[ds + '_rank'] = rows
    print('picked', pick['folk'], pick['pop909'])

    # ---- 3/4. render + timeline
    timeline = {'about': __doc__.split('\n\n')[0], 'cfg': {'J': cfg.J, 'bpm': cfg.bpm, 'L': cfg.L, 'delta': cfg.delta,
                'alto': list(cfg.alto)}, 'noise': bench.NOISE, 'classes': {
                'good': sorted(CONSONANT), 'harsh': sorted(bench.HARSH), 'fourth': [5], 'silent': 'Duet silent'},
                'val_check': check, 'val_overlap_top3': agg, 'harsh_causes_doc': CAUSE_DOC,
                'val_harsh_by_cause': agg_causes, 'songs': []}
    for ds, sid in (('twinkle', None), ('folk', pick['folk']), ('pop909', pick['pop909'])):
        s, r = next((s, r) for s, r in table[ds] if sid is None or s['id'] == sid)
        full = s
        sc, n_kept = crop(s, POP_CROP) if ds == 'pop909' else (s, None)
        if ds == 'twinkle':
            slug, title = 'twinkle', s.get('title', 'Twinkle')
        elif ds == 'folk':
            slug, title = 'folk_' + slugify(s['title']), s['title']
        else:
            num = s['id'].split('/')[-1]
            slug, title = f'pop909_{num}', f'POP909 #{num}'
        heard_full = bench.heard(full)['sop']             # same order and timing as full['sop']
        heard_sc = [[on, min(off, sc['K']), m] for on, off, m in heard_full if on < sc['K']]
        assert [n[:2] for n in heard_sc] == [n[:2] for n in sc['sop']]
        files, t0, T_b, count_in = render(sc, r, cfg, slug, a.wav_dir)
        segs = {n: segments(sc, r[n], cfg.J) for n in SYSTEMS}
        Dk = (cfg.L + cfg.delta) / T_b
        caus = {n: harsh_causes(sc, heard_sc, segs[n], Dk, cfg.alto) for n in SYSTEMS}
        entry = {
            'slug': slug, 'title': title, 'id': s['id'], 'dataset': ds, 'tonic': s['tonic'], 'tonic_name': NAMES[s['tonic']],
            'bpm': sc.get('bpm', cfg.bpm), 'M': s['M'], 'K': sc['K'], 'K_full': full['K'],
            'cropped': ds == 'pop909' and full['K'] > sc['K'],
            'crop_note': (f'first {sc["K"]} of {full["K"]} beats; systems ran on the whole heard song (causal, so '
                          f'identical to running on the crop)') if ds == 'pop909' else None,
            'scale': song_scale(sc)[0], 'J': cfg.J, 'audio': {'t0_s': t0, 'T_b_s': T_b, 'count_in_beats': count_in, 'files': files,
                                  'duet_level': DUET_LEVEL, 'click_gain': CLICK_GAIN},
            'bars': bar_labels(sc),
            'her_true': [[on, off, m] for on, off, m in sc['sop']],
            'heard': [[on, off, mh, int(mh != mt)] for (on, off, mh), (_, _, mt) in zip(heard_sc, sc['sop'])],
            'systems': {},
            'overlap_top3': overlap(segs),
        }
        for n in SYSTEMS:
            hk = {b.k: b.h for b in r[n]}
            e = {'h': [hk.get(k, -1) for k in range(sc['K'])],
                 'segments': [[st, en, k, c, iv, cz] for (_, k, st, en, c, iv), cz in zip(segs[n], caus[n])],
                 'shares': {k: v for k, v in shares(segs[n]).items() if k != 'dur'},
                 'harsh_by_cause': cause_shares(segs[n], caus[n]),
                 'out_of_scale': out_of_scale(sc, r[n], cfg.J)}
            if entry['cropped']:
                e['shares_full_song'] = {k: v for k, v in shares(segments(full, r[n], cfg.J)).items() if k != 'dur'}
            entry['systems'][n] = e
        entry['segment_fields'] = ['start_beat', 'end_beat', 'k', 'class', 'interval_class (her - Duet) % 12, null if silent',
                                   'harsh cause (see harsh_causes_doc), null unless harsh']
        timeline['songs'].append(entry)

        # per-beat print for writing the descriptions
        print(f"\n=== {slug}: {title}  tonic {NAMES[s['tonic']]}  K {sc['K']}  M {s['M']}")
        for n in SYSTEMS:
            print('  ', n, {k: round(v, 3) for k, v in entry['systems'][n]['shares'].items()},
                  {k: round(v, 3) for k, v in entry['systems'][n]['harsh_by_cause'].items()},
                  'out-of-scale %.2f' % entry['systems'][n]['out_of_scale'])
        print('   overlap', {k: (round(v, 3) if v is not None else None) for k, v in entry['overlap_top3'].items()})
        lab = entry['bars']
        for k in range(cfg.J - 1, sc['K']):
            hers = [(on, m, f) for (on, off, m), (_, _, _, f) in zip(sc['sop'], entry['heard']) if on < k + 1 and off > k]
            her_s = ' '.join(nname(m) + ('*' if f else '') for _, m, f in hers)
            cols = []
            for n in SYSTEMS:
                cs = [(c[0] if c != 'harsh' else 'H' + cz[0].upper() + ('n' if cz.endswith('ctx') else ''))
                      for st, en, kk, c, iv, cz in entry['systems'][n]['segments'] if kk == k]
                cols.append(f"{nname(entry['systems'][n]['h'][k]):4s}{' '.join(cs):9s}")
            print(f'   k{k:3d} {lab[k]:5s} her {her_s:18s} | ' + ' | '.join(cols))

    timeline['selection'] = {
        'rule': 'among val songs of 16-40 s at the bench bpm (POP909 cropped to its first %d beats), the one where '
                'the classes of policy, gru and retrieval disagree on the largest share of her melody time' % POP_CROP,
        'folk_top': pick['folk_rank'][:8], 'pop909_top': pick['pop909_rank'][:8]}
    with open(os.path.join(OUT, 'timeline.json'), 'w') as fh:
        json.dump(timeline, fh, default=float)
    print('\nfiles')
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        d = f'{duration(p):6.1f}s' if f.endswith('.mp3') else '       '
        print(f'  {p}  {d}  {os.path.getsize(p) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
