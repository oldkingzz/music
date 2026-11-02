"""Listening example: Greensleeves (synthetic singer) with the rule harmony and the learned one.

    .venv/bin/python harmony/make_examples.py

Uses the real rehearsal path: singer.render -> follow.analyse -> follow.offline_notes, so the
harmony planners see exactly what they would see from a recording. Writes into
harmony/examples/ (about 6 MB):
  greensleeves_rule.wav, greensleeves_learned.wav       singer + machine voice, mono 24 kHz
  harmony_rule.wav, harmony_learned.wav                 machine voice only
  greensleeves_compare.musicxml                         melody + both harmony lines (3/4 score)
  plans.json                                            both plans (note, r, dur, midi, h)
and harmony/figures/greensleeves_pianoroll.png. Nothing is played.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import singer                                                        # noqa: E402
from dsp import render_voice, write_wav                              # noqa: E402
from follow import analyse, offline_notes, plan_harmony              # noqa: E402
from metrics import line_metrics, pool, rates                         # noqa: E402
from model import plan_harmony_learned                               # noqa: E402

EX = os.path.join(HERE, 'examples')
FIG = os.path.join(HERE, 'figures')
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def name(m):
    return f'{NAMES[m % 12]}{m // 12 - 1}'


def musicxml(path, plans):
    """Melody + harmony lines on the beat grid of singer.GREENSLEEVES (score notes)."""
    from music21 import meter, note, stream, tempo
    sc = stream.Score()
    for label, hs in [('Melody (singer)', [m for _, _, m in singer.GREENSLEEVES])] + plans:
        part = stream.Part()
        part.partName = label
        part.append(tempo.MetronomeMark(number=84))
        part.append(meter.TimeSignature('3/4'))
        part.append(note.Rest(quarterLength=2))                      # pickup bar padded to 3/4
        for (beat, length, _), m in zip(singer.GREENSLEEVES, hs):
            part.append(note.Note(int(m), quarterLength=length))
        sc.insert(0, part)
    sc.write('musicxml', fp=path)


def pianoroll(path, notes, plans, x_len):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, sharey=True)
    colors = {'Rule (third/sixth below)': '#eb6834', 'Learned CRF (Bach-trained)': '#2a78d6'}
    for ax, (label, plan) in zip(axes, plans):
        for nt in notes:
            ax.add_patch(plt.Rectangle((nt['on'], nt['midi'] - 0.4), nt['off'] - nt['on'], 0.8,
                                       color='#52514e', lw=0))
        for p in plan:
            ax.add_patch(plt.Rectangle((p['r'], p['h'] - 0.4), p['dur'], 0.8, color=colors[label], lw=0))
        ax.set_title(f'{label}  (grey = singer)', loc='left', fontsize=11)
        ax.set_ylim(min(p['h'] for _, pl in plans for p in pl) - 2, max(n['midi'] for n in notes) + 2)
        ax.set_xlim(0, x_len)
        ax.set_ylabel('MIDI pitch')
        ax.grid(axis='y', color='#e4e3df', lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(EX, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    score = [{'on': a, 'off': b, 'midi': m, 'kind': 'attack'} for a, b, m in singer.score_notes()]
    x, info = singer.render(singer.score_notes())
    notes = offline_notes(analyse(x))                                # what rehearsal would see
    plans = {}
    for label, fn in (('rule', plan_harmony), ('learned', plan_harmony_learned)):
        plan, key = fn(notes)
        plans[label] = {'plan': plan, 'key': key}
        h = render_voice([(p['r'], p['dur'], p['h']) for p in plan], len(x))
        mix = 0.8 * x + h
        mix *= 0.9 / np.max(np.abs(mix))
        write_wav(os.path.join(EX, f'greensleeves_{label}.wav'), mix)
        write_wav(os.path.join(EX, f'harmony_{label}.wav'), h * (0.9 / np.max(np.abs(h))))
        s = [p['midi'] for p in plan]
        plans[label]['metrics'] = rates(pool([line_metrics(s, [p['h'] for p in plan])]))
        plans[label]['lines'] = ' '.join(f"{name(p['midi'])}/{name(p['h'])}" for p in plan)
    # score-grid versions for the MusicXML export (same planners, exact score notes)
    sp_rule, _ = plan_harmony(score)
    sp_learn, _ = plan_harmony_learned(score)
    musicxml(os.path.join(EX, 'greensleeves_compare.musicxml'),
             [('Rule harmony', [p['h'] for p in sp_rule]), ('Learned harmony', [p['h'] for p in sp_learn])])
    pianoroll(os.path.join(FIG, 'greensleeves_pianoroll.png'), notes,
              [('Rule (third/sixth below)', plans['rule']['plan']),
               ('Learned CRF (Bach-trained)', plans['learned']['plan'])], len(x) / 24000)
    summary = {'n_score_notes': len(score), 'n_offline_notes': len(notes),
               'offline_midi_matches_score': [nt['midi'] for nt in notes] == [nt['midi'] for nt in score],
               **{k: {kk: vv for kk, vv in v.items() if kk != 'plan'} for k, v in plans.items()},
               'score_grid': {'rule': [p['h'] for p in sp_rule], 'learned': [p['h'] for p in sp_learn]}}
    with open(os.path.join(EX, 'plans.json'), 'w') as fh:
        json.dump({'summary': summary, 'rule': plans['rule']['plan'], 'learned': plans['learned']['plan']},
                  fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
