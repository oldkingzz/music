"""Streaming machine voice: dsp.render_voice, rendered block by block.

Notes are scheduled at absolute sample indices on the output timeline and rendered as the
blocks go by, with the oscillator phase carried across block boundaries, so the stream is
sample-for-sample the same waveform render_voice() would produce offline for the same notes
(checked in tests/test_voice.py). Two things only a live renderer has to handle:

* a note scheduled for a sample that has already been emitted starts at the first free
  sample (it is late; the caller logs how late);
* a note's length can be shortened after it started (the next note arrived, or a false
  phrase entry was taken back). The release then starts no earlier than the current render
  position, so the envelope never jumps (no click).
"""
import numpy as np

from dsp import SR, midi_to_hz

ATTACK = 0.025
RELEASE = 0.06
TAIL = 0.08                       # render_voice renders dur + 0.08 s


class _Note:
    __slots__ = ('i0', 'dur', 'midi', 'level', 'fsum', 'done', 'tag')

    def __init__(self, i0, dur, midi, level, tag):
        self.i0, self.dur, self.midi, self.level, self.tag = i0, float(dur), float(midi), level, tag
        self.fsum = 0.0           # running sum of instantaneous frequency = phase * SR / 2pi
        self.done = 0             # samples of this note already rendered


class StreamingVoice:
    def __init__(self, level=0.22):
        self.level = level
        self.notes = []           # active notes (scheduled or sounding)
        self.pos = 0              # next output sample index to be rendered
        self.by_tag = {}

    def schedule(self, start_sample, dur, midi, tag=None):
        """Add a note. Returns the sample it will actually start at (>= start_sample)."""
        i0 = max(int(start_sample), self.pos)
        n = _Note(i0, max(dur, 0.0), midi, self.level, tag)
        self.notes.append(n)
        if tag is not None:
            self.by_tag[tag] = n
        return i0

    def set_dur(self, tag, dur):
        """Shorten/lengthen a note. Returns the duration actually applied."""
        n = self.by_tag.get(tag)
        if n is None:
            return None
        rendered = n.done / SR
        dur = float(dur)
        if dur <= n.dur:                           # shorten: release cannot start in the past,
            n.dur = min(n.dur, max(dur, rendered))  # and shortening never lengthens
        elif rendered <= n.dur:                    # lengthen only while the release has not begun
            n.dur = dur
        return n.dur

    def release_now(self, tag):
        n = self.by_tag.get(tag)
        if n is None:
            return None
        return self.set_dur(tag, max(0.0, (self.pos - n.i0) / SR))

    def active(self):
        return len(self.notes)

    def render(self, n_samples):
        """Render the next n_samples of the output timeline (from self.pos)."""
        s0, s1 = self.pos, self.pos + n_samples
        out = np.zeros(n_samples)
        keep = []
        for nt in self.notes:
            length = int((nt.dur + TAIL) * SR)
            end = nt.i0 + length
            a, b = max(s0, nt.i0), min(s1, end)
            if b > a:
                idx = np.arange(a - nt.i0, b - nt.i0)
                if idx[0] != nt.done:                 # phase continuity guard (should never fire)
                    raise RuntimeError('voice rendered out of order')
                t = idx / SR
                vib = 0.15 * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.15) / 0.2, 0.0, 1.0)
                cs = nt.fsum + np.cumsum(midi_to_hz(nt.midi + vib))
                nt.fsum = float(cs[-1])
                ph = 2 * np.pi * cs / SR
                y = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
                env = np.clip(t / ATTACK, 0.0, 1.0) * np.clip((nt.dur + RELEASE - t) / RELEASE, 0.0, 1.0)
                out[a - s0:b - s0] += y * env * nt.level
                nt.done = b - nt.i0
            if end > s1:
                keep.append(nt)
            elif nt.tag is not None and self.by_tag.get(nt.tag) is nt:
                del self.by_tag[nt.tag]
        self.notes = keep
        self.pos = s1
        return out.astype(np.float32)
