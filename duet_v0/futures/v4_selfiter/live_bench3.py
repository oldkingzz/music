"""Any bench3 system live: phone speaker -> mic -> Duet's note in the headphones (LIVE_NOTE.md step 3).

Bench3Duet is live_mic.LiveDuet (through policy_live.PolicyDuet, for its online note segmentation) with
the decision moved from beats to her notes, the way bench3 scores a system: each time the tracker finds a
new note of hers (an onset, or 30 ms on a new pitch), the system's step(i, onset_beats, heard_midi,
prev_dur) is called once and Duet starts its note at once, holding it until her next note or until she
stops. Beat 0 is her first sung note (Twinkle starts on beat 0), so nothing has to be lined up by hand;
tempo, key and meter are given. No click is played. Duet's voice goes to the output device only.

Run from the repo root:
  .venv/bin/python futures/v4_selfiter/live_bench3.py --devices
  .venv/bin/python futures/v4_selfiter/live_bench3.py --selftest
  .venv/bin/python futures/v4_selfiter/live_bench3.py --system r2_mix_v2 --in "<mic name>" --out "<headphones name>"
Each live run writes runs_mini/live/<HHMMSS>_<system>/run.wav (L mic, R Duet) and notes.json; never overwrites.
"""
import argparse
import importlib
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ABS, 'futures', 'panel'))
sys.path.insert(0, os.path.join(ABS, 'duet_beat'))
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
import live_mic as lm  # noqa: E402
from policy_live import PolicyDuet  # noqa: E402
from follow import FeatureTracker  # noqa: E402
from voice import StreamingVoice  # noqa: E402
from dsp import HOP, SR, midi_to_hz  # noqa: E402

SYSTEMS = ('r2_mix_v2', 'm1_filter_v5lo', 'third_below', 'r3_style_mix_v3')
OUT_ROOT = os.path.join(ABS, 'runs_mini', 'live')


def build(name, train):
    return importlib.import_module('systems.' + name).build(train)


_tt = np.arange(4096) / 4096 * 2 * np.pi
REED_PEAK = float(np.max(np.abs(np.sin(_tt) + 0.45 * np.sin(3 * _tt) + 0.22 * np.sin(5 * _tt) + 0.1 * np.sin(7 * _tt)))) * 1.06


class ReedVoice(StreamingVoice):
    """StreamingVoice with make_demos3.reed's sound (the offline demos Vincent judged fine in Phase B):
    odd harmonics 1, .45, .22, .1, a 4.5 Hz 6 % tremolo, no pitch vibrato, 60 ms attack, 50 ms release.
    `level` is each note's peak, as reed()'s level is after its normalisation."""
    ATTACK, RELEASE = 0.06, 0.05

    def render(self, n_samples):
        s0, s1 = self.pos, self.pos + n_samples
        out = np.zeros(n_samples)
        keep = []
        for nt in self.notes:
            end = nt.i0 + int((nt.dur + self.RELEASE) * SR)
            a, b = max(s0, nt.i0), min(s1, end)
            if b > a:
                t = np.arange(a - nt.i0, b - nt.i0) / SR
                ph = 2 * np.pi * float(midi_to_hz(nt.midi)) * t
                y = np.sin(ph) + 0.45 * np.sin(3 * ph) + 0.22 * np.sin(5 * ph) + 0.1 * np.sin(7 * ph)
                y *= 1 + 0.06 * np.sin(2 * np.pi * 4.5 * t)
                env = np.clip(t / self.ATTACK, 0.0, 1.0) * np.clip((nt.dur + self.RELEASE - t) / self.RELEASE, 0.0, 1.0)
                out[a - s0:b - s0] += y * env * nt.level / REED_PEAK
                nt.done = b - nt.i0
            if end > s1:
                keep.append(nt)
            elif nt.tag is not None and self.by_tag.get(nt.tag) is nt:
                del self.by_tag[nt.tag]
        self.notes = keep
        self.pos = s1
        return out.astype(np.float32)


class Bench3Duet(PolicyDuet):
    """One bench3 system answering each of her notes as it is heard."""

    def __init__(self, system, name, tonic, M, K, bpm, first_beat=0.0, listen_only=False, monitor=0.0,
                 hold_gap=0.25, grid=0.25, voice='reed', log=print):
        # LiveDuet's own __init__ builds the chord HMM planner, which a bench3 system does not use
        self.system, self.name = system, name
        self.tonic, self.M, self.K = tonic, M, K
        self.cfg = types.SimpleNamespace(bpm=bpm)
        self.T_b = 60.0 / bpm
        self.first_beat, self.count_in = first_beat, 0
        self.t0 = None                                       # set by her first note
        self.Li = self.Lo = 0.0
        self.listen_only, self.monitor, self.hold_gap, self.log = listen_only, monitor, hold_gap, log
        self.grid = grid                                     # snap onsets within 0.1 beat of this grid (0: off)
        self.ft = FeatureTracker()
        self.voice = ReedVoice(level=0.42) if voice == 'reed' else StreamingVoice(level=0.22)
        self.n = 0
        self.ft_t, self.ft_m, self.ft_v, self.on_t = [], [], [], []
        self.seg_i, self.notes, self.cur, self.gap = 0, [], None, 0
        self.events = []                                     # one per her note answered
        self.answered = None                                 # start time of the note last answered
        self.tag, self.last_voiced = None, None
        self.ended = False
        self.late, self.proc_ms = [], []
        system.start({'tonic': tonic, 'M': M, 'K': K, 'pos': [k % M for k in range(K)]})

    def heard(self):
        return self.ft_t[-1] if self.ft_t else -1.0

    def done(self):
        return self.ended or (self.t0 is not None and self.heard() > self.t0 + (self.K + 1) * self.T_b)

    def process(self, x):
        c0 = time.perf_counter()
        x = np.asarray(x, float)
        i = self.ft.step(x)
        t = self.ft.t(i) - self.Li
        self.ft_t.append(t)
        self.ft_m.append(np.nan_to_num(self.ft.m[-1]))
        self.ft_v.append(self.ft.v[-1])
        nov = self.ft.nov
        if self.ft.v[-1] and nov[-1] >= 0.5 and (len(nov) < 2 or nov[-2] < 0.5):
            self.on_t.append(t)
        if self.ft.v[-1]:
            self.last_voiced = t
        self._segment()
        self._react(t)
        y = self.voice.render(HOP) + self.monitor * x
        self.n += HOP
        self.proc_ms.append((time.perf_counter() - c0) * 1000)
        return y

    def _segment(self):
        """PolicyDuet._segment with one change: a new pitch must last NEW_FRAMES frames (60 ms, not 30 ms)
        before it starts a new note, so a singer's glide or a dip at a note's end is not a note of its own."""
        on_set = set(np.round(np.asarray(self.on_t[-50:]), 4)) if self.on_t else set()
        while self.seg_i < len(self.ft_t):
            i = self.seg_i
            t, m, v = self.ft_t[i], self.ft_m[i], self.ft_v[i]
            self.seg_i += 1
            if not v:
                self.gap += 1
                if self.cur is not None and self.gap > 3:
                    self._close()
                continue
            self.gap = 0
            q = int(round(m))
            onset = round(t, 4) in on_set
            c = self.cur
            if c is None:
                self.cur = [t, t, {q: 1}, None, t, 0]
                continue
            cur_m = max(c[2], key=c[2].get)
            if onset and t - c[0] > 0.06:
                self._close()
                self.cur = [t, t, {q: 1}, None, t, 0]
                continue
            if q != cur_m:
                if c[3] == q:
                    c[5] += 1
                else:
                    c[3], c[4], c[5] = q, t, 1
                if c[5] >= self.NEW_FRAMES:
                    start = c[4]
                    c[1] = start - 0.01
                    self._close()
                    self.cur = [start, t, {q: self.NEW_FRAMES}, None, t, 0]
                    continue
            else:
                c[3], c[5] = None, 0
            c[1] = t
            c[2][q] = c[2].get(q, 0) + 1

    NEW_FRAMES = 6

    def _snap(self, x, floor=False):
        """Bench melodies are written on a grid; the GRU's on-beat input needs onsets exactly on it."""
        g = self.grid
        if not g:
            return x
        q = round(x / g) * g
        if floor:
            q = max(q, g)
        return q if abs(x - q) <= 0.1 else x

    def _react(self, t):
        c = self.cur
        need = 12 if self.t0 is None else 3                  # the first note syncs beat 0: 120 ms, not a click
        if c is not None and sum(c[2].values()) >= need and c[0] != self.answered and not self.ended:
            self.answered = c[0]
            midi = max(c[2], key=c[2].get)
            if self.t0 is None:
                self.t0 = c[0] - self.first_beat * self.T_b
            onset_raw = (c[0] - self.t0) / self.T_b
            onset = self._snap(onset_raw)
            if onset >= self.K:
                self.ended = True
            else:
                prev = self.notes[-1] if self.notes else None
                prev_dur = None if not self.events or prev is None else self._snap((prev[1] - prev[0]) / self.T_b, True)
                i = len(self.events)
                c1 = time.perf_counter()
                h = int(self.system.step(i, float(onset), int(midi), prev_dur))
                step_ms = (time.perf_counter() - c1) * 1000
                if self.tag is not None:
                    self.voice.release_now(self.tag)
                    self.tag = None
                if h >= 0 and not self.listen_only:
                    self.tag = ('n', i)
                    self.voice.schedule(self.voice.pos, 60.0, h, tag=self.tag)
                lag = (t - c[0]) * 1000
                self.events.append({'i': i, 'onset_beats': round(onset, 3), 'onset_raw': round(onset_raw, 3),
                                    'her_heard': int(midi), 'duet': h,
                                    'prev_dur': None if prev_dur is None else round(prev_dur, 3),
                                    'detect_ms': round(lag, 1), 'step_ms': round(step_ms, 2)})
                bar, beat = divmod(int(np.floor(onset + 1e-6)), self.M)
                self.log(f'{bar + 1:3d}.{beat + 1}  she {lm.note_name(int(midi)):4s} -> Duet '
                         f'{lm.note_name(h) if h >= 0 else "silent":6s}  heard {lag:3.0f} ms after her onset')
        if self.tag is not None and (self.cur is None and self.last_voiced is not None
                                     and t - self.last_voiced > self.hold_gap):
            self.voice.release_now(self.tag)                 # she stopped: Duet stops too
            self.tag = None
        if self.ended and self.tag is not None:
            self.voice.release_now(self.tag)
            self.tag = None

    def summary(self):
        lags = [e['detect_ms'] for e in self.events]
        return {'system': self.name, 'tonic': self.tonic, 'M': self.M, 'K': self.K, 'bpm': self.cfg.bpm,
                't0': self.t0, 'Li': self.Li, 'Lo': self.Lo, 'monitor': self.monitor,
                'notes_answered': len(self.events), 'sung': sum(e['duet'] >= 0 for e in self.events),
                'detect_ms_median': float(np.median(lags)) if lags else None,
                'block_ms_median': float(np.median(self.proc_ms)) if self.proc_ms else None,
                'block_ms_max': float(np.max(self.proc_ms)) if self.proc_ms else None,
                'events': self.events}


def pick(kind, want):
    """Device index whose name contains `want` and has channels of this kind. The iPhone microphone is used
    only when asked for by name (Vincent, 2026-11-01: the iPad plays the test file, the iPhone listens)."""
    import sounddevice as sd
    key = 'max_input_channels' if kind == 'in' else 'max_output_channels'
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and want.lower() in d['name'].lower():
            if kind == 'in' and 'iphone' in d['name'].lower() and 'iphone' not in want.lower():
                continue
            return i
    raise SystemExit(f'no {kind}put device whose name contains {want!r}; run --devices')


def selftest(names, bpm=80.0, Li=0.012):
    """Rendered Twinkle (C, 80 bpm) block by block through Bench3Duet; compare with the batch call."""
    from audio import render_her
    from songs import twinkle
    song = twinkle()
    true = sorted(song['sop'], key=lambda n: n[0])
    meta = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'pos': [b['pos'] for b in song['beats']]}
    train = bench3.train_sets()
    T_b, lead = 60.0 / bpm, 0.5
    for name in names:
        sysm = build(name, train)
        sysm.start(meta)
        batch, prev = [], None
        for i, (on, off, m) in enumerate(true):
            batch.append(int(sysm.step(i, float(on), int(m), None if prev is None else float(prev[1] - prev[0]))))
            prev = (on, off)
        import soundfile as sf
        phone, sr = sf.read(os.path.join(ABS, 'futures', 'live_test', 'phone', 'twinkle_C_80bpm_countin.wav'),
                            dtype='float32', always_2d=True)
        assert sr == SR, sr
        for style in ('clean', 'vibrato', 'phone file'):
            eng = Bench3Duet(sysm, name, song['tonic'], song['M'], song['K'], bpm, log=lambda s: None)
            eng.Li = Li
            her = phone.mean(1) if style == 'phone file' else render_her(song, lead, T_b, style)
            x = np.concatenate([np.zeros(int(round(Li * SR))), her, np.zeros(SR)])
            for b in range(len(x) // HOP):
                eng.process(x[b * HOP:(b + 1) * HOP])
            ev = eng.events
            match = []                                       # each true note -> the event that starts within 0.2 beat
            for (on, _, m), h in zip(true, batch):
                near = [e for e in ev if abs(e['onset_beats'] - on) < 0.2]
                match.append((m, h, near[0] if near else None))
            found = sum(e is not None for _, _, e in match)
            pitch_ok = sum(e is not None and e['her_heard'] == m for m, _, e in match)
            same = sum(e is not None and e['duet'] == h for _, h, e in match)
            s = eng.summary()
            print(f'{name:16s} {style:10s} found {found:2d}/{len(true)} (+{len(ev) - found} extra)  pitch right {pitch_ok:2d}  '
                  f'same note as batch {same:2d}/{len(true)}  '
                  f'heard {s["detect_ms_median"]:.0f} ms after onset  block {s["block_ms_median"]:.2f}/{s["block_ms_max"]:.2f} ms',
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--system', choices=SYSTEMS, default='r2_mix_v2')
    ap.add_argument('--song', default='twinkle', help='label for the log (the phone file you play)')
    ap.add_argument('--bpm', type=float, default=80.0)
    ap.add_argument('--key', default='C', help='major key (tonic) of the phone file')
    ap.add_argument('--meter', type=int, default=4)
    ap.add_argument('--beats', type=int, default=48, help='beats of the song (Twinkle: 48)')
    ap.add_argument('--first-beat', type=float, default=0.0, help='beat of her first sung note (Twinkle: 0)')
    ap.add_argument('--in', dest='dev_in', default=None, help='microphone: part of its name')
    ap.add_argument('--out', dest='dev_out', default=None, help='headphones: part of their name')
    ap.add_argument('--monitor', type=float, default=0.0, help='also play the mic into the headphones (0..1)')
    ap.add_argument('--listen', action='store_true', help='mic check: Duet decides but stays silent')
    ap.add_argument('--devices', action='store_true', help='list sound devices and exit')
    ap.add_argument('--selftest', action='store_true', help='offline check, no sound card')
    a = ap.parse_args()

    if a.devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if a.selftest:
        selftest(('r2_mix_v2', 'm1_filter_v5lo', 'third_below'))
        return
    if not a.dev_in or not a.dev_out:
        raise SystemExit('give --in and --out (run --devices to see the names)')
    dev = (pick('in', a.dev_in), pick('out', a.dev_out))
    print(f'loading {a.system} ...', flush=True)
    sysm = build(a.system, bench3.train_sets())
    eng = Bench3Duet(sysm, a.system, lm.KEYS[a.key], a.meter, a.beats, a.bpm, first_beat=a.first_beat,
                     listen_only=a.listen, monitor=a.monitor)
    run_dir = os.path.join(OUT_ROOT, time.strftime('%H%M%S') + '_' + a.system + ('_listen' if a.listen else ''))
    os.makedirs(run_dir)                                     # fails rather than reuse a folder
    import sounddevice as sd
    print(f'input: {sd.query_devices(dev[0])["name"]} | output: {sd.query_devices(dev[1])["name"]}\n'
          f'{a.song}, {a.key} major, {a.meter}/4, {a.bpm:.0f} BPM; beat 0 = her first note. '
          f'Start the phone now. Ctrl-C stops.', flush=True)
    lm.run_live(eng, device=dev, record=os.path.join(run_dir, 'run.wav'), meter=True)
    s = eng.summary()
    s['song'], s['in'], s['out'] = a.song, sd.query_devices(dev[0])['name'], sd.query_devices(dev[1])['name']
    with open(os.path.join(run_dir, 'notes.json'), 'x') as fh:
        json.dump(s, fh, indent=1, default=float)
    print(f"\nanswered {s['notes_answered']} notes, Duet sang {s['sung']}; heard her notes "
          f"{s['detect_ms_median'] or 0:.0f} ms after onset (median); block {s['block_ms_median'] or 0:.2f} ms median\n"
          f'saved {run_dir}/run.wav (L mic, R Duet) and notes.json')


if __name__ == '__main__':
    main()
