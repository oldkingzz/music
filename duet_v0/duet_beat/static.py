"""Static Duet: she sings the whole take first, then Duet harmonises it with everything heard.

Nothing has to be guessed ahead of time, so the chord path is the single most likely one given
the WHOLE melody (Viterbi over the chord HMM, past and future beats), and Duet's notes are then
chosen for those chords exactly as the 'oracle' system does. This is the no-latency fallback
from the PDR ("record, then answer") and shows what the same model does without the deadline.

  .venv/bin/python duet_beat/static.py        # score it on the Bach test set, Twinkle, arpeggios
"""
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import ChordHMM  # noqa: E402
from planner import Config, ScoreSource, run, to_obs  # noqa: E402
from theory import N_CHORDS  # noqa: E402


def viterbi_chords(song, model, cfg, source=None):
    """Most likely chord per beat given every beat's melody. Returns a list of chord indices."""
    source = source or ScoreSource(song)
    t, M, K = song['tonic'], song['M'], song['K']
    pos = [b['pos'] for b in song['beats']]
    ll = [model.loglik(to_obs(*source.full(k), t, cfg.mode), cfg.mode, cfg.lam) for k in range(K)]
    delta = np.log(model.init) + ll[0]
    back = []
    for k in range(1, K):
        A = np.log(model.trans(pos[k], M))
        tot = delta[:, None] + A
        b = np.argmax(tot, axis=0)
        back.append(b)
        delta = tot[b, np.arange(N_CHORDS)] + ll[k]
    z = [int(np.argmax(delta))]
    for b in reversed(back):
        z.append(int(b[z[-1]]))
    return z[::-1]


def run_static(song, model, cfg, source=None):
    """Duet's beats for a recorded take: Viterbi chords, then notes as in the oracle system."""
    z = viterbi_chords(song, model, cfg, source)
    told = dict(song, beats=[dict(b, z=zz) for b, zz in zip(song['beats'], z)])
    beats = run(told, model, replace(cfg, q_min=0.0), 'oracle', source=source)
    for b, bt in zip(beats, song['beats']):
        b.z_true = bt['z']                                    # score against the real chords again
    return beats


def main():
    import json
    from data import load
    from metrics import song_counts, summarize
    from songs import arpeggios, twinkle
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    cfg = Config(**base)
    out = {}
    for name, songs in (('bach_test', d['test']), ('twinkle', [twinkle()]), ('arpeggios', [arpeggios()])):
        row = {}
        for sysn in ('live', 'static'):
            cs = []
            for s in songs:
                c = replace(cfg, bpm=s.get('bpm', cfg.bpm))
                beats = run(s, model, c, 'hmm') if sysn == 'live' else run_static(s, model, c)
                cs.append(song_counts(s, beats, cfg.J))
            row[sysn] = summarize(cs)
        out[name] = row
        print(f"{name:10s} live: chord {row['live']['m1']:.3f} consonant {row['live']['m3']:.3f}   "
              f"static: chord {row['static']['m1']:.3f} consonant {row['static']['m3']:.3f}")
    with open(os.path.join(HERE, 'out', 'static.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
