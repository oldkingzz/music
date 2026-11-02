"""Listening demos for bench v3: her line (formant 'ah' voice, louder, left of centre) + Duet's line
(clarinet-like odd-harmonic reed, softer, right of centre). Duet runs exactly as in bench3: it hears her
notes with the 10 % mishearing, one at a time, and holds its note until her next onset.
MP3 is written by piping raw PCM into lame (no wav left behind).

  .venv/bin/python futures/v3_nolatency/make_demos3.py third_below m3_seq_v5
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench3 as B3  # noqa: E402  (also puts duet_beat on sys.path)
import importlib  # noqa: E402
from audio import render_clean  # noqa: E402
from data import load  # noqa: E402
from dsp import SR, midi_to_hz  # noqa: E402
from songs import twinkle  # noqa: E402

OUT = os.path.join(HERE, 'audio')
LAME = '/opt/homebrew/bin/lame'
MAX_S = 38.0


def cut(song, K):
    """First K beats of a song (systems are causal, so their notes there are unchanged)."""
    s = dict(song)
    s['sop'] = [[on, min(off, K), m] for on, off, m in song['sop'] if on < K]
    s['beats'] = song['beats'][:K]
    s['K'] = K
    return s


def songs():
    val = {s['id']: s for s in load()['split']['val']}
    tw = twinkle()
    out = [('twinkle', tw, tw['bpm'])]
    for cid, bpm in (('bach/bwv415', 84), ('bach/bwv308', 80)):
        out.append((cid.split('/')[1], val[cid], bpm))
    res = []
    for name, s, bpm in out:
        T_b = 60.0 / bpm
        K = min(s['K'], int(MAX_S / T_b) - 1)
        res.append((name, cut(s, K) if K < s['K'] else s, T_b))
    return res


def reed(notes, n, level):
    """notes: (start_s, end_s, midi). Clarinet-like: odd harmonics, slow attack, gentle tremolo."""
    y = np.zeros(n)
    for st, en, m in notes:
        i0, i1 = int(st * SR), min(n, int(en * SR))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / SR
        ph = 2 * np.pi * midi_to_hz(m) * t
        s = np.sin(ph) + 0.45 * np.sin(3 * ph) + 0.22 * np.sin(5 * ph) + 0.1 * np.sin(7 * ph)
        s *= 1 + 0.06 * np.sin(2 * np.pi * 4.5 * t)
        dur = t[-1]
        env = np.clip(t / 0.06, 0, 1) * np.clip((dur - t) / 0.05, 0, 1)
        y[i0:i1] += s * env
    return y * level / (np.max(np.abs(y)) + 1e-9)


def render(song, hs, T_b, path):
    true = sorted(song['sop'], key=lambda x: x[0])
    t0 = 0.4
    her = render_clean([(t0 + on * T_b, t0 + off * T_b - 0.03, m) for on, off, m in true], level=1.0)
    dn = []
    for i, (on, off, m) in enumerate(true):
        if hs[i] < 0:
            continue
        end = true[i + 1][0] if i + 1 < len(true) else off
        dn.append((t0 + on * T_b, t0 + end * T_b - 0.02, hs[i]))
    n = len(her)
    du = reed(dn, n, level=0.42)
    st = np.stack([0.8 * her + 0.35 * du, 0.45 * her + 0.75 * du], 1)
    st *= 0.9 / (np.max(np.abs(st)) + 1e-9)
    pcm = (np.clip(st, -1, 1) * 32767).astype('<i2').tobytes()
    subprocess.run([LAME, '--quiet', '-r', '--little-endian', '--signed', '--bitwidth', '16',
                    '-s', str(SR / 1000), '-m', 's', '-b', '128', '-', path], input=pcm, check=True)
    return n / SR


def main(names):
    train = B3.train_sets()
    meta = []
    for sysname in names:
        system = importlib.import_module(f'systems.{sysname}').build(train)
        for song_name, song, T_b in songs():
            hs = B3.run_song(system, song)
            a = B3.song_stats(song, hs)
            path = os.path.join(OUT, f'{song_name}_{sysname}.mp3')
            secs = render(song, hs, T_b, path)
            meta.append({'file': path, 'song': song_name, 'system': sysname,
                         'music': a['music'] / (a['music_d'] or float('nan')),
                         'double': a['double'] / a['d'], 'seconds': round(secs, 1), 'beats': song['K'],
                         'bpm': round(60 / T_b)})
            print(json.dumps(meta[-1]))
    with open(os.path.join(OUT, 'demos.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
