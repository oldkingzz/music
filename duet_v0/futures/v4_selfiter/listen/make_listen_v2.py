"""Blind A/B listening test, v2 (Phase B §3). Same as make_listen.py, with one fix found before anyone listened:
v1 drew bwv264 for an oracle-vs-m1 pair (and its repeat) where the two sides differ in 1 of 12 notes, i.e. a coin
flip. v2 draws songs with a neutral rule that favours neither side: the two systems of a pair must differ in at
least 3 of the clip's notes, and a tied-vs-re-attacked clip must contain at least 2 ties to merge.

Every system runs on the FULL song (bench3.run_song); only then is the first K = min(K, 16) beats cut and rendered
with make_demos3's render (re-attacked) or the tied copy in make_listen.py. Bach val chorales (not bwv308 / bwv415)
and Twinkle only.

  cd $ABS/futures/v4_selfiter/listen && $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 \
      $ABS/.venv/bin/python $ABS/futures/v4_selfiter/listen/make_listen_v2.py

Writes v2/audio/pNN_L.mp3 + v2/audio/pNN_R.mp3, v2/ab.html (no system names) and v2/key.json. Refuses if v2/ exists.
"""
import importlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_listen as L  # noqa: E402  (also puts futures/v3_nolatency on sys.path)
from make_listen import D, bench3, load, twinkle  # noqa: E402

V2 = os.path.join(HERE, 'v2')
OUT = os.path.join(V2, 'audio')
SEED = 1103
M1, TB, M3, OR = 'm1_filter_v5lo', 'third_below', 'm3_seq_v5', 'm4_oracle'


def ties(hs):
    return sum(1 for a, b in zip(hs, hs[1:]) if a >= 0 and a == b)


def main():
    if os.path.exists(V2):
        sys.exit('refusing: v2/ exists')
    os.makedirs(OUT)
    rng = random.Random(SEED)
    train = bench3.train_sets()
    systems = {n: importlib.import_module(f'systems.{n}').build(train) for n in (TB, M1, M3, OR)}
    tw = twinkle()
    val = [s for s in load()['split']['val'] if s['id'] not in L.SKIP]
    rng.shuffle(val)
    runs = {}

    def hs_of(song, name):
        if (song['id'], name) not in runs:
            runs[(song['id'], name)] = bench3.run_song(systems[name], song)    # FULL song, never the cut
        return runs[(song['id'], name)]

    def clip_len(song):
        return len(D.cut(song, min(song['K'], L.KMAX))['sop'])

    def differ(song, a, b):
        n = clip_len(song)
        return sum(x != y for x, y in zip(hs_of(song, a)[:n], hs_of(song, b)[:n]))

    used, groups = set(), {}
    need = [('oracle_vs_m1', 4, lambda s: differ(s, OR, M1) >= 3),
            ('m1_vs_third_below', 3, lambda s: differ(s, M1, TB) >= 3),
            ('m1_vs_m3', 3, lambda s: differ(s, M1, M3) >= 3),
            ('tied_vs_reattack', 2, lambda s: ties(hs_of(s, M1)[:clip_len(s)]) >= 2)]
    for g, k, ok in need:
        groups[g] = []
        for s in val:
            if len(groups[g]) < k and s['id'] not in used and ok(s):
                groups[g].append(s)
                used.add(s['id'])
        if len(groups[g]) < k:
            sys.exit(f'not enough songs for {g}')
    assert differ(tw, M1, TB) >= 3 and differ(tw, M1, M3) >= 3 and ties(hs_of(tw, M1)[:clip_len(tw)]) >= 2
    pairs = ([('m1_vs_third_below', s, (M1, 'reattack'), (TB, 'reattack')) for s in [tw] + groups['m1_vs_third_below']]
             + [('oracle_vs_m1', s, (OR, 'reattack'), (M1, 'reattack')) for s in groups['oracle_vs_m1']]
             + [('m1_vs_m3', s, (M1, 'reattack'), (M3, 'reattack')) for s in [tw] + groups['m1_vs_m3']]
             + [('tied_vs_reattack', s, (M1, 'tied'), (M1, 'reattack')) for s in [tw] + groups['tied_vs_reattack']])
    order = list(range(len(pairs)))
    rng.shuffle(order)
    seq = [dict(src=j, swap=rng.random() < 0.5, repeat_of=None) for j in order]
    rep_pos = rng.choice([p for p, x in enumerate(seq) if pairs[x['src']][0] in ('m1_vs_third_below', 'oracle_vs_m1')
                          and p <= len(seq) - 5])
    ins = rng.randint(rep_pos + 5, len(seq))
    seq.insert(ins, dict(src=seq[rep_pos]['src'], swap=not seq[rep_pos]['swap'], repeat_of=rep_pos + 1))

    key, page = [], []
    for p, item in enumerate(seq, 1):
        group, song, x, y = pairs[item['src']]
        sides = (y, x) if item['swap'] else (x, y)
        K = min(song['K'], L.KMAX)
        cut = D.cut(song, K)
        ent = {'pair': p, 'group': group, 'song': song['id'], 'beats': K, 'repeat_of': item['repeat_of'],
               'notes_differ': differ(song, x[0], y[0]) if x[0] != y[0] else 0,
               'ties': ties(hs_of(song, M1)[:len(cut['sop'])]) if group == 'tied_vs_reattack' else None}
        for side, (sysname, variant) in zip('LR', sides):
            full = hs_of(song, sysname)
            hs = full[:len(cut['sop'])]
            path = os.path.join(OUT, f'p{p:02d}_{side}.mp3')
            secs = (L.render_tied if variant == 'tied' else D.render)(cut, hs, L.T_B, path)
            c_nd, c_dbl = L.nd(cut, hs)
            s_nd, s_dbl = L.nd(song, full)
            ent[side] = {'system': sysname, 'variant': variant, 'clip_nd': c_nd, 'clip_double': c_dbl,
                         'song_nd': s_nd, 'song_double': s_dbl, 'seconds': round(secs, 1), 'n_held': len(cut['sop']),
                         'notes': hs}
        key.append(ent)
        title = 'Twinkle' if song['id'] == 'twinkle' else 'Bach ' + song['id'].split('/')[-1]
        page.append({'pair': p, 'title': title})
        print(json.dumps({k: v for k, v in ent.items() if k not in ('L', 'R')}),
              ent['L']['system'], ent['L']['variant'], ent['L']['clip_nd'], '|',
              ent['R']['system'], ent['R']['variant'], ent['R']['clip_nd'], flush=True)
    with open(os.path.join(V2, 'key.json'), 'x') as fh:
        json.dump({'version': 2, 'seed': SEED, 'bpm': L.BPM, 'beats_max': L.KMAX, 'pairs': key,
                   'selection': 'pairs differ in >= 3 clip notes; tied clips have >= 2 ties (neutral to both sides)',
                   'code': 'L2-<16 letters L/R = preferred side per pair>[-<pair><L|R><tags c=clashing b=boring '
                           'j=jumpy r=robotic> joined by .]'}, fh, indent=1)
    tpl = open(os.path.join(HERE, 'ab_template.html')).read()
    html = tpl.replace('/*PAIRS*/[]', json.dumps(page)).replace("'duet_ab_v1'", "'duet_ab_v2'").replace("'L1-'", "'L2-'")
    with open(os.path.join(V2, 'ab.html'), 'x') as fh:
        fh.write(html)


if __name__ == '__main__':
    main()
