"""Blind A/B listening test for v4_selfiter Phase B (§3): 16 forced-choice pairs, 12 s clips at 80 bpm.

Every system runs on the FULL song (bench3.run_song, so m4_oracle's lookup still finds the song); only then is
the first K = min(K, 16) beats cut and rendered with make_demos3's own render (re-attacked) or a copy of it that
merges consecutive equal Duet notes into one held note (tied). Bach chorales from the val split (not bwv308 /
bwv415) and Twinkle only.

  cd $ABS/futures/v4_selfiter/listen && $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 \
      $ABS/.venv/bin/python $ABS/futures/v4_selfiter/listen/make_listen.py

Writes audio/pNN_L.mp3 + audio/pNN_R.mp3, ab.html (no system names) and key.json (which side is which).
Refuses to run if any of them exists.
"""
import importlib
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABS = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import make_demos3 as D  # noqa: E402  (importing is safe; never call D.main() or D.songs())
import bench3  # noqa: E402
from data import load  # noqa: E402
from songs import twinkle  # noqa: E402
from theory import CONSONANT, CT  # noqa: E402

BPM = 80
T_B = 60.0 / BPM
KMAX = int(12 / T_B)
OUT = os.path.join(HERE, 'audio')
SKIP = {'bach/bwv308', 'bach/bwv415'}
SEED = 1102


def render_tied(song, hs, T_b, path):
    """Copy of make_demos3.render; consecutive equal Duet notes become one held note before D.reed."""
    true = sorted(song['sop'], key=lambda x: x[0])
    t0 = 0.4
    her = D.render_clean([(t0 + on * T_b, t0 + off * T_b - 0.03, m) for on, off, m in true], level=1.0)
    dn, last = [], None
    for i, (on, off, m) in enumerate(true):
        if hs[i] < 0:
            last = None
            continue
        end = true[i + 1][0] if i + 1 < len(true) else off
        if last == i - 1 and dn[-1][2] == hs[i]:
            dn[-1] = (dn[-1][0], t0 + end * T_b - 0.02, hs[i])
        else:
            dn.append((t0 + on * T_b, t0 + end * T_b - 0.02, hs[i]))
        last = i
    n = len(her)
    du = D.reed(dn, n, level=0.42)
    st = np.stack([0.8 * her + 0.35 * du, 0.45 * her + 0.75 * du], 1)
    st *= 0.9 / (np.max(np.abs(st)) + 1e-9)
    pcm = (np.clip(st, -1, 1) * 32767).astype('<i2').tobytes()
    subprocess.run([D.LAME, '--quiet', '-r', '--little-endian', '--signed', '--bitwidth', '16',
                    '-s', str(D.SR / 1000), '-m', 's', '-b', '128', '-', path], input=pcm, check=True)
    return n / D.SR


def nd(song, hs):
    """music_nd of one (possibly cut) song, exactly as rescore_nd.py counts it; also the doubling share."""
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
    return round(num / den, 4) if den else None, round(dbl / den, 4) if den else None


def main():
    for p in ('ab.html', 'key.json'):
        if os.path.exists(os.path.join(HERE, p)):
            sys.exit(f'refusing: {p} exists')
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    val = [s for s in load()['split']['val'] if s['id'] not in SKIP]
    pick = [val[i] for i in rng.sample(range(len(val)), 12)]
    tw = twinkle()
    A, B, C, Dd = pick[0:3], pick[3:7], pick[7:10], pick[10:12]
    M1, TB, M3, OR = 'm1_filter_v5lo', 'third_below', 'm3_seq_v5', 'm4_oracle'
    pairs = ([('m1_vs_third_below', s, (M1, 'reattack'), (TB, 'reattack')) for s in [tw] + A]
             + [('oracle_vs_m1', s, (OR, 'reattack'), (M1, 'reattack')) for s in B]
             + [('m1_vs_m3', s, (M1, 'reattack'), (M3, 'reattack')) for s in [tw] + C]
             + [('tied_vs_reattack', s, (M1, 'tied'), (M1, 'reattack')) for s in [tw] + Dd])
    order = list(range(len(pairs)))
    rng.shuffle(order)
    seq = [dict(src=j, swap=rng.random() < 0.5, repeat_of=None) for j in order]
    rep_pos = rng.choice([p for p, x in enumerate(seq) if pairs[x['src']][0] in ('m1_vs_third_below', 'oracle_vs_m1')
                          and p <= len(seq) - 5])
    ins = rng.randint(rep_pos + 5, len(seq))
    seq.insert(ins, dict(src=seq[rep_pos]['src'], swap=not seq[rep_pos]['swap'], repeat_of=rep_pos + 1))

    systems, runs = {}, {}
    train = bench3.train_sets()
    for name in (TB, M1, M3, OR):
        systems[name] = importlib.import_module(f'systems.{name}').build(train)
    key, page = [], []
    for p, item in enumerate(seq, 1):
        group, song, x, y = pairs[item['src']]
        sides = (y, x) if item['swap'] else (x, y)
        K = min(song['K'], KMAX)
        cut = D.cut(song, K)
        ent = {'pair': p, 'group': group, 'song': song['id'], 'beats': K, 'repeat_of': item['repeat_of']}
        for side, (sysname, variant) in zip('LR', sides):
            rk = (song['id'], sysname)
            if rk not in runs:
                runs[rk] = bench3.run_song(systems[sysname], song)           # FULL song, never the cut
            hs = runs[rk][:len(cut['sop'])]
            path = os.path.join(OUT, f'p{p:02d}_{side}.mp3')
            if os.path.exists(path):
                sys.exit(f'refusing: {path} exists')
            secs = (render_tied if variant == 'tied' else D.render)(cut, hs, T_B, path)
            c_nd, c_dbl = nd(cut, hs)
            s_nd, s_dbl = nd(song, runs[rk])
            ent[side] = {'system': sysname, 'variant': variant, 'clip_nd': c_nd, 'clip_double': c_dbl,
                         'song_nd': s_nd, 'song_double': s_dbl, 'seconds': round(secs, 1),
                         'n_held': len(cut['sop'])}
        key.append(ent)
        title = 'Twinkle' if song['id'] == 'twinkle' else 'Bach ' + song['id'].split('/')[-1]
        page.append({'pair': p, 'title': title})
        print(json.dumps(ent), flush=True)
    with open(os.path.join(HERE, 'key.json'), 'x') as fh:
        json.dump({'seed': SEED, 'bpm': BPM, 'beats_max': KMAX, 'pairs': key,
                   'code': 'L1-<16 letters L/R = preferred side per pair>[-<pair><L|R><tags c=clashing b=boring '
                           'j=jumpy r=robotic> joined by .]'}, fh, indent=1)
    tpl = open(os.path.join(HERE, 'ab_template.html')).read()
    with open(os.path.join(HERE, 'ab.html'), 'x') as fh:
        fh.write(tpl.replace('/*PAIRS*/[]', json.dumps(page)))


if __name__ == '__main__':
    main()
