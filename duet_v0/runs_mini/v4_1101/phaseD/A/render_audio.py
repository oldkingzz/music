"""Phase D audio (PROGRAM §5.3), run v4_1101: 4 systems x 3 songs = 12 mp3 clips + index.md.

Systems: r2_mix_v2 (presentable champion), r3_style_mix_v3 (best score, not presentable), m1_filter_v5lo (old model),
third_below (simple baseline). Songs: Twinkle and the first two Bach val chorales in sorted id order other than
bach/bwv308 and bach/bwv415 (earlier demos). Licences: Bach chorales (public domain) and Twinkle ONLY.

Same pattern as futures/v4_selfiter/listen/make_listen.py (Phase B): every system runs on the FULL song with
bench3.run_song (she is heard with the benchmark's 10 % mishearing, one note at a time; a system never runs on a cut
song); only then is the first K = min(K, 40) beats cut (80 bpm, T_b = 0.75 s, about 30 s) and rendered with
make_demos3's own render, re-attacked notes (Phase B chose no tied renders). make_demos3 is only imported:
D.main() / D.songs() are never called (they write audio/demos.json).

  cd $ABS/runs_mini/v4_1101/phaseD/A && $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/hbrun.py pD_A_render 600 \
      $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/phaseD/A/render_audio.py

Writes report/audio/<song>_<system>.mp3 (12 files), report/audio/index.md and phaseD/A/render_log.json. Refuses to
start if any of them exists (never overwrites); every number is computed before the first file is written.
"""
import hashlib
import importlib
import json
import os
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v4_1101')
OUT = os.path.join(RUN, 'report', 'audio')
INDEX = os.path.join(OUT, 'index.md')
LOG = os.path.join(RUN, 'phaseD', 'A', 'render_log.json')
SYSDIR = os.path.join(ABS, 'futures', 'v3_nolatency', 'systems')

sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import make_demos3 as D  # noqa: E402  (importing is safe; never call D.main() or D.songs())
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
from data import load  # noqa: E402
from songs import twinkle  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

BPM = 80
T_B = 60.0 / BPM                              # 0.75 s per beat
KMAX = 40                                     # clip = first min(K, 40) beats = at most 30 s of music
SKIP = {'bach/bwv308', 'bach/bwv415'}         # earlier demos (make_demos3)
SYSTEMS = (('r2_mix_v2', 'presentable champion'),
           ('r3_style_mix_v3', 'best score, not presentable'),
           ('m1_filter_v5lo', 'old model m1'),
           ('third_below', 'simple baseline'))
# sha256 prefixes recorded in state_R4.json (r2, r3), state_R2.json (r2_style_hmm_v1) and tools/legacy_systems.sha256
SHA = {'r2_mix_v2.py': 'b481d9156f3c', 'r3_style_mix_v3.py': 'b79ea9bf01c4', 'r2_style_hmm_v1.py': 'a37e0f1b3168',
       'm1_filter_v5lo.py': '70bae2a359cc', 'third_below.py': '9d1928320916', 'm3_seq_v4.py': 'bd62540fa354',
       'm3_seq_full.pt': '35441332b9b4', 'm3_seq_big.pt': '8d1d6a848438', 'm3_seq_s1.pt': '9a311c698b58'}
VAL_HEADLINE = {'r2_mix_v2': 0.6444, 'r3_style_mix_v3': 0.6532, 'm1_filter_v5lo': 0.6206, 'third_below': 0.5054}


def nd(song, hs):
    """music_nd and double of one (possibly cut) song, exactly as rescore_nd.py counts them: her TRUE notes from beat
    bench3.J = 4, beats with a chord label; music_nd = consonant with her AND a tone of the true chord AND not her
    note or its octave; double = her note or its octave."""
    true = sorted(song['sop'], key=lambda n: n[0]); z = [b['z'] for b in song['beats']]; t = song['tonic']
    num = den = dbl = 0.0
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
    return (num / den, dbl / den) if den else (None, None)


def f3(x):
    return '-' if x is None else f'{x:.3f}'


def index_md(songs, rows, total_bytes):
    out = ['# Duet audio clips: champion vs old model vs baseline (Phase D, run v4_1101)', '',
           'How to listen: use headphones; her voice (the "ah" singer) leans left and Duet\'s clarinet leans right.',
           'Play the four files of one song back to back and follow the clarinet: it is Duet\'s real-time harmony '
           'line under her melody.', '',
           '| file | song | system | role | beats | seconds | clip music_nd | clip double | song music_nd '
           '| clip notes unlike m1 |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        sec = '-' if r['seconds'] is None else f"{r['seconds']:.1f}"
        out.append(f"| {r['file']} | {r['song']} | {r['system']} | {r['role']} | {r['beats']} | {sec} "
                   f"| {f3(r['clip_nd'])} | {f3(r['clip_double'])} | {f3(r['song_nd'])} "
                   f"| {r['unlike_m1']} / {r['her_notes_clip']} |")
    out += ['', '## Songs', '']
    for sn, s in songs:
        what = 'Twinkle Twinkle Little Star (traditional)' if sn == 'twinkle' else \
            f"Bach chorale {s['id']} (val split, public domain)"
        K = min(s['K'], KMAX)
        n = len(D.cut(s, K)['sop'])
        out.append(f"- {sn}: {what}; {s['M']} beats per bar, {s['K']} beats in all; the clip is beats 0-{K - 1} "
                   f"({K} beats, {n} of her notes).")
    out += ['', '## What the numbers mean', '',
            '- Every system ran on the FULL song exactly as in bench3: Duet hears her notes one at a time with the '
            'benchmark\'s 10 % mishearing, never sees the chord labels, and holds each note until her next onset. '
            'The clip was cut afterwards: the first min(song beats, 40) beats at 80 bpm (0.75 s per beat, about '
            '30 s), re-attacked notes (Phase B chose no tied renders).',
            '- clip music_nd: share of her singing time in the clip, from beat 4 on (Duet may listen first), where '
            'Duet\'s note is consonant with her TRUE note AND a tone of the TRUE chord AND not her own note or its '
            'octave (rescore_nd\'s definition). clip double: share of the same time where Duet sings her note or '
            'its octave. song music_nd: the same score over the whole song (this one song, not the dataset).',
            '- clip notes unlike m1: how many of Duet\'s notes in the clip differ from m1_filter_v5lo\'s, out of '
            'her notes in the clip.',
            '- For reference, val headline music_nd over all four datasets: ' + ', '.join(
                f'{k} {v:.4f}' for k, v in VAL_HEADLINE.items()) + '; told the chord (m4_oracle) 0.8388.',
            '- All numbers are val (this bundle has no test split) and every system was tuned on val.',
            '- Licences: audio of Bach chorales (public domain) and Twinkle only; never folk, pop or pop909.']
    if total_bytes is not None:
        out.append(f'- {len(rows)} mp3 files, {total_bytes / 1e6:.2f} MB in all (128 kbit/s stereo).')
    out.append(f"- Made by {LOG.replace(ABS, '$ABS').replace('render_log.json', 'render_audio.py')} "
               f"on {time.strftime('%Y-%m-%d %H:%M %Z')}; per-clip details and Duet's notes in "
               f"{LOG.replace(ABS, '$ABS')}.")
    return '\n'.join(out) + '\n'


def main():
    t_start = time.time()
    # 1. songs (licence: Bach chorales and Twinkle only)
    val = {s['id']: s for s in load()['split']['val']}
    bach_ids = sorted(i for i in val if i not in SKIP)[:2]
    assert len(bach_ids) == 2 and all(i.startswith('bach/') for i in bach_ids), bach_ids
    tw = twinkle()
    assert tw['id'] == 'twinkle', tw.get('id')
    songs = [('twinkle', tw)] + [(i.split('/')[-1], val[i]) for i in bach_ids]
    print('songs:', [s['id'] for _, s in songs], flush=True)

    # 2. refuse before any work if an output exists; lame present; no M1_/M3_/DUET_HIST knobs; system files unchanged
    plan = [(sn, s, name, role, os.path.join(OUT, f'{sn}_{name}.mp3')) for sn, s in songs for name, role in SYSTEMS]
    for p in [x[-1] for x in plan] + [INDEX, LOG]:
        if os.path.exists(p):
            sys.exit(f'refusing: {p} exists')
    if not os.path.isdir(OUT):
        sys.exit(f'missing directory {OUT} (mkdir -p it first)')
    if not os.access(D.LAME, os.X_OK):
        sys.exit(f'lame not found at {D.LAME}')
    knobs = sorted(k for k in os.environ if k.startswith(('M1_', 'M3_', 'DUET_HIST')))
    if knobs:
        sys.exit(f'refusing: environment knobs set {knobs}')
    sha = {}
    for fn, pre in SHA.items():
        with open(os.path.join(SYSDIR, fn), 'rb') as fh:
            sha[fn] = hashlib.sha256(fh.read()).hexdigest()
        if not sha[fn].startswith(pre):
            sys.exit(f'refusing: {fn} sha256 {sha[fn][:12]} != recorded {pre}')
    print('sha256 ok:', {k: v[:12] for k, v in sha.items()}, flush=True)

    # 3. build every system once; run it on the FULL song; cut; score the clip and the whole song
    train = bench3.train_sets()
    built = {name: importlib.import_module(f'systems.{name}').build(train) for name, _ in SYSTEMS}
    print(f'built in {time.time() - t_start:.0f} s', flush=True)
    full = {(sn, name): bench3.run_song(built[name], s) for sn, s in songs for name, _ in SYSTEMS}
    rows = []
    for sn, s, name, role, path in plan:
        K = min(s['K'], KMAX)
        cut = D.cut(s, K)
        n = len(cut['sop'])
        hs = full[(sn, name)][:n]
        c_nd, c_dbl = nd(cut, hs)
        s_nd, s_dbl = nd(s, full[(sn, name)])
        unlike = sum(a != b for a, b in zip(hs, full[(sn, 'm1_filter_v5lo')][:n]))
        rows.append({'file': os.path.basename(path), 'path': path, 'song': sn, 'song_id': s['id'], 'system': name,
                     'role': role, 'beats': K, 'song_beats': s['K'], 'her_notes_clip': n, 'seconds': None,
                     'bytes': None, 'clip_nd': c_nd, 'clip_double': c_dbl, 'song_nd': s_nd, 'song_double': s_dbl,
                     'unlike_m1': unlike, 'duet_clip': hs, '_cut': cut})
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                          if k not in ('path', 'duet_clip', '_cut')}), flush=True)
    index_md(songs, rows, None)                                  # format once now, so a typo fails before any write

    # 4. render (new files only), then the log and the index (mode 'x')
    for r in rows:
        if os.path.exists(r['path']):
            sys.exit(f"refusing: {r['path']} exists")
        r['seconds'] = round(D.render(r['_cut'], r['duet_clip'], T_B, r['path']), 2)
        r['bytes'] = os.path.getsize(r['path'])
        print(f"wrote {r['file']} {r['seconds']} s {r['bytes']} bytes", flush=True)
    total = sum(r['bytes'] for r in rows)
    keep = [{k: v for k, v in r.items() if k != '_cut'} for r in rows]
    with open(LOG, 'x') as fh:
        json.dump({'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'bpm': BPM, 'T_b': T_B, 'beats_max': KMAX,
                   'songs': [s['id'] for _, s in songs], 'skip': sorted(SKIP), 'render': 'make_demos3.render (re-attacked)',
                   'sha256': sha, 'clips': keep, 'total_bytes': total,
                   'note': 'val only, tuned on val (no test split in this bundle)'}, fh, indent=1)
    with open(INDEX, 'x') as fh:
        fh.write(index_md(songs, rows, total))
    print(f'total {total} bytes ({total / 1e6:.2f} MB) in {len(rows)} files; {time.time() - t_start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
