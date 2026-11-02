"""Negative tests for review.py and keep.py (Phase A): the checks must FAIL on systems known to break them.
Writes nothing.

  cd $ABS/runs_mini/v4_1101/tools && $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 600 \
      $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review  # noqa: E402
import keep  # noqa: E402
from review import bench3  # noqa: E402


class Peek:
    """Cheats through the call stack: sings a third below her NEXT note (only possible by peeking at run_song)."""
    def start(self, meta):
        pass

    def step(self, i, onset, midi, prev_dur):
        notes = sys._getframe(1).f_locals['notes']
        return notes[i + 1][2] - 4 if i + 1 < len(notes) else midi - 4


class Above:
    def start(self, meta):
        pass

    def step(self, i, onset, midi, prev_dur):
        return min(midi + 4, 77)


c = review.causal(Peek())
print('peek: causal ok =', c['ok'], 'mismatches', len(c['mismatches']), 'of', c['checks'], '(must be False)')
g = review.grep('m4_oracle')
print('m4_oracle: grep ok =', g['ok'], 'fails', sorted({f.get('token', f.get('why')) for f in g['fails']}), '(must be False)')
d = review.rng_check(Above())
print('above: range ok =', d['ok'], 'above_share', round(d['above_share'], 3), '(must be False)')
v = keep.verdict('m3_seq_v5', 'm1_filter_v5lo', gate=True)
print('keep m3_seq_v5 vs m1: pass =', v['pass'], 'failed', v['failed'], '(must be False)')
v = keep.verdict('m1_filter_v5lo', 'm1_filter_v5lo', gate=True)
print('keep m1 vs itself: pass =', v['pass'], 'failed', v['failed'], '(must be False: needs +0.005)')
print('bench3 J =', bench3.J)
