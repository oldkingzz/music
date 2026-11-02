"""Context for the Phase D audio clips (read-only, no audio): are the two Bach clip songs typical?

For every Bach val chorale in data.load()['split']['val']: whole-song and first-min(K, 40)-beat music_nd and double of
r2_mix_v2, r3_style_mix_v3, m1_filter_v5lo and third_below, counted exactly as render_audio.py / rescore_nd.py count
them (every system on the FULL song with bench3.run_song); how often the champion beats m1; and the time-pooled bach
music_nd as a check against the dataset-level numbers (r2_mix_v2 0.6853, m1_filter_v5lo 0.66547).

  cd $ABS/runs_mini/v4_1101/phaseD/A && $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/hbrun.py pD_A_context 600 \
      $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/phaseD/A/clip_context.py

Prints a table; writes phaseD/A/clip_context.json (mode 'x', refuses if it exists). Val only, tuned on val.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_audio as R  # noqa: E402  (constants and imports only; its main() is not called)

OUTJ = os.path.join(R.RUN, 'phaseD', 'A', 'clip_context.json')
CHAMP, OLD = 'r2_mix_v2', 'm1_filter_v5lo'


def parts(song, hs):
    """(num, den, dbl) of rescore_nd's music_nd over one (possibly cut) song."""
    true = sorted(song['sop'], key=lambda n: n[0]); z = [b['z'] for b in song['beats']]; t = song['tonic']
    num = den = dbl = 0.0
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(R.bench3.J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            den += ov
            if h >= 0:
                iv = abs(m - h) % 12
                num += ov * (iv in R.CONSONANT and iv != 0 and (h - t) % 12 in R.CT[z[k]])
                dbl += ov * (iv == 0)
    return num, den, dbl


def wtl(rows, key):
    d = [r[CHAMP][key] - r[OLD][key] for r in rows]
    return {'champion_better': sum(x > 1e-9 for x in d), 'tie': sum(abs(x) <= 1e-9 for x in d),
            'm1_better': sum(x < -1e-9 for x in d), 'mean_diff': round(sum(d) / len(d), 4)}


def main():
    if os.path.exists(OUTJ):
        sys.exit(f'refusing: {OUTJ} exists')
    names = [n for n, _ in R.SYSTEMS]
    train = R.bench3.train_sets()
    built = {n: importlib.import_module(f'systems.{n}').build(train) for n in names}
    val = sorted(R.load()['split']['val'], key=lambda s: s['id'])
    assert all(s['id'].startswith('bach/') for s in val)
    rows, pool = [], {n: [0.0, 0.0] for n in names}
    for s in val:
        K = min(s['K'], R.KMAX)
        cut = R.D.cut(s, K)
        n_clip = len(cut['sop'])
        r = {'id': s['id'], 'beats': s['K'], 'earlier_demo': s['id'] in R.SKIP}
        for nm in names:
            hs = R.bench3.run_song(built[nm], s)                     # FULL song, never the cut
            a, b, c = parts(s, hs)
            ca, cb, cc = parts(cut, hs[:n_clip])
            pool[nm][0] += a
            pool[nm][1] += b
            r[nm] = {'song_nd': a / b, 'song_double': c / b, 'clip_nd': ca / cb, 'clip_double': cc / cb}
        rows.append(r)
    summary = {'songs': len(rows),
               'song_level': wtl(rows, 'song_nd'), 'clip_level': wtl(rows, 'clip_nd'),
               'pooled_bach_music_nd': {n: round(v[0] / v[1], 5) for n, v in pool.items()},
               'rank_of_clip_songs_by_song_diff': {}}
    order = sorted(rows, key=lambda r: r[CHAMP]['song_nd'] - r[OLD]['song_nd'], reverse=True)
    for i, r in enumerate(order, 1):
        if r['id'] in ('bach/bwv140.7', 'bach/bwv245.26'):
            summary['rank_of_clip_songs_by_song_diff'][r['id']] = f'{i} of {len(order)} (1 = champion gains most)'
    print(f"{'id':16s} {'beats':>5s}  song: {'r2':>6s} {'r3':>6s} {'m1':>6s} {'3rd':>6s} {'r2-m1':>7s}   "
          f"clip: {'r2':>6s} {'m1':>6s} {'r2-m1':>7s}  dbl r2/m1")
    for r in rows:
        g = lambda nm, k: r[nm][k]  # noqa: E731
        print(f"{r['id']:16s} {r['beats']:5d}  song: {g(CHAMP, 'song_nd'):6.3f} {g('r3_style_mix_v3', 'song_nd'):6.3f} "
              f"{g(OLD, 'song_nd'):6.3f} {g('third_below', 'song_nd'):6.3f} "
              f"{g(CHAMP, 'song_nd') - g(OLD, 'song_nd'):+7.3f}   clip: {g(CHAMP, 'clip_nd'):6.3f} {g(OLD, 'clip_nd'):6.3f} "
              f"{g(CHAMP, 'clip_nd') - g(OLD, 'clip_nd'):+7.3f}  {g(CHAMP, 'clip_double'):.3f}/{g(OLD, 'clip_double'):.3f}"
              + ('  (earlier demo)' if r['earlier_demo'] else ''))
    print(json.dumps(summary, indent=1))
    with open(OUTJ, 'x') as fh:
        json.dump({'what': __doc__.split('\n\n')[0], 'summary': summary, 'rows': rows,
                   'note': 'val only, tuned on val (no test split in this bundle)'}, fh, indent=1)


if __name__ == '__main__':
    main()
