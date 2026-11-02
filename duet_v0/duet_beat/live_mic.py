"""Duet live on the MacBook: she sings into the mic on a click, Duet harmonises in real time.

The same causal loop as planner.run (system 'hmm'), but fed 10 ms at a time from a sounddevice
duplex stream instead of a whole rendered song. Everything lives on one acoustic timeline:
  an output sample written at stream counter n is heard at (n + Lo) / SR,
  an input sample read at stream counter n was sung at (n - Li) / SR,
so Duet's note for beat k is written at counter tau_k * SR - Lo and must be decided before that
block is rendered, from frames that have already arrived. Nothing reads the future.

LiveDuet.process() is the whole engine (one block in, one block out); the sounddevice wrapper and
the offline check (--selftest) only feed it. Use headphones: the click and Duet's voice must not
reach the mic.

Run from the repo root:
  .venv/bin/python duet_beat/live_mic.py --selftest              # offline: synthetic Twinkle
  .venv/bin/python duet_beat/live_mic.py --listen --beats 16     # mic only, Duet silent
  .venv/bin/python duet_beat/live_mic.py --beats 48              # the demo (live)
  .venv/bin/python duet_beat/live_mic.py --static --beats 32     # sing first, Duet answers after
Writes duet_beat/out/live/<stamp>.{json,wav} (log, and a stereo wav: L mic, R click + Duet).
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'live'))
from data import load  # noqa: E402
from model import ChordHMM  # noqa: E402
from planner import Config, Planner, _prior_chain, to_obs  # noqa: E402
from songs import slow_harmony  # noqa: E402
from theory import ALTO, CT, N_CHORDS, PC_NAMES, chord_name  # noqa: E402
from dsp import HOP, SR, write_wav  # noqa: E402
from follow import FeatureTracker  # noqa: E402
from voice import StreamingVoice  # noqa: E402

OUT = os.path.join(HERE, 'out', 'live')
KEYS = {n: i for i, n in enumerate(PC_NAMES)} | {'Db': 1, 'D#': 3, 'Gb': 6, 'G#': 8, 'A#': 10}


def note_name(m):
    return f'{PC_NAMES[m % 12]}{m // 12 - 1}' if m is not None and m >= 0 else '-'


def load_model(style):
    m = ChordHMM().fit(load()['split']['train'])
    return slow_harmony(m) if style == 'slow' else m


class LiveDuet:
    def __init__(self, model, cfg, tonic=0, M=4, K=48, count_in=4, Li=0.0, Lo=0.0, lead=0.5,
                 listen_only=False, monitor=True, repeat=True, log=print):
        self.model, self.cfg, self.tonic, self.M, self.K = model, cfg, tonic, M, K
        self.T_b = 60.0 / cfg.bpm
        self.count_in = count_in
        self.t0 = lead + count_in * self.T_b                # acoustic time of beat 0
        self.Li, self.Lo = Li, Lo
        self.listen_only, self.monitor, self.repeat, self.log = listen_only, monitor, repeat, log
        self.ft = FeatureTracker()
        self.voice = StreamingVoice(level=0.22)
        self.planner = Planner(tonic, cfg, model)
        self.pos = [k % M for k in range(K)]
        self.n = 0                                           # stream counter of the next block
        self.ft_t, self.ft_m, self.ft_v, self.on_t = [], [], [], []
        self.alpha, self.pi = [], []                         # full-beat filter, as in planner.run
        self.beats = []                                      # one dict per decided beat
        self.joined, self.q_prev, self.h_last, self.her_last, self.stopped = False, 0.0, None, None, False
        self.last_note = None                                # (tag, start counter, h) of the note sounding
        self.late = []                                       # decisions made too late (samples)
        self.grid = []                                       # her melody, 4 points per beat (repetition rule)
        self.proc_ms = []

    # ------------------------------------------------------------------ timeline
    def tau(self, k):
        return self.t0 + k * self.T_b

    def heard(self):
        return self.ft_t[-1] if self.ft_t else -1.0

    def done(self):
        return self.heard() > self.tau(self.K) + 0.3

    # ------------------------------------------------------------------ observations (audio.AudioSource, online)
    def _obs(self, a, b):
        t = np.asarray(self.ft_t)
        sel = (t >= a) & (t < b) & np.asarray(self.ft_v, bool)
        idx = np.where(sel)[0]
        vec = np.zeros(12)
        if len(idx) == 0:
            return vec, None
        r = np.round(np.asarray(self.ft_m)[idx]).astype(int)
        fr = HOP / SR / self.T_b
        for m in r:
            vec[m % 12] += fr
        pc = int(np.argmax(vec))
        return vec, int(np.median(r[(r % 12) == pc]))

    def _ll(self, vec, mid):
        return self.model.loglik(to_obs(vec, mid, self.tonic, self.cfg.mode), self.cfg.mode, self.cfg.lam)

    def _drift_ms(self, m=4):
        ts = np.asarray(self.on_t[-m:])
        if len(ts) < m:
            return None
        g = 0.5 * self.T_b
        e = (ts - self.t0) - np.round((ts - self.t0) / g) * g
        return float(np.median(e) * 1000.0)

    # ------------------------------------------------------------------ one block
    def process(self, x):
        """x: HOP mic samples read at counter self.n. Returns (HOP,) mono output for the same counter."""
        c0 = time.perf_counter()
        i = self.ft.step(np.asarray(x, float))
        t = self.ft.t(i) - self.Li
        self.ft_t.append(t)
        self.ft_m.append(np.nan_to_num(self.ft.m[-1]))
        self.ft_v.append(self.ft.v[-1])
        nov = self.ft.nov
        if self.ft.v[-1] and nov[-1] >= 0.5 and (len(nov) < 2 or nov[-2] < 0.5):
            self.on_t.append(t)
        # close the full-beat filter for every beat now completely heard
        while len(self.alpha) < self.K and t >= self.tau(len(self.alpha) + 1):
            j = len(self.alpha)
            p = self.model.init if j == 0 else self.model.predict(self.alpha[j - 1], self.pos[j], self.M)
            self.pi.append(p)
            self.alpha.append(self.model.update(p, self._ll(*self._obs(self.tau(j), self.tau(j + 1)))))
        # decide every beat whose note must be written in the next block (or is already late)
        horizon_end = self.n + 2 * HOP + self.cfg.delta * SR
        while len(self.beats) < self.K and self.tau(len(self.beats)) * SR - self.Lo * SR < horizon_end:
            self._decide(len(self.beats))
        y = self.voice.render(HOP) + self._click(HOP)
        self.n += HOP
        self.proc_ms.append((time.perf_counter() - c0) * 1000)
        return y

    def _click(self, n):
        y = np.zeros(n)
        for k in range(-self.count_in, self.K):
            i0 = int(round((self.tau(k) - self.Lo) * SR)) - self.n
            L = int(0.025 * SR)
            if i0 + L <= 0 or i0 >= n:
                continue
            tt = (np.arange(L) / SR)
            f = 1760.0 if k % self.M == 0 else 1320.0
            s = (0.35 if k % self.M == 0 else 0.22) * np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.006)
            a, b = max(i0, 0), min(i0 + L, n)
            y[a:b] += s[a - i0:b - i0]
        return y

    # ------------------------------------------------------------------ the decision for beat k (planner.run, 'hmm')
    def _decide(self, k):
        cfg, M, model = self.cfg, self.M, self.model
        h_t = self.heard()
        when = (h_t - self.t0) / self.T_b                      # what has been heard, in beats
        jstar = math.floor(when)
        frac = when - jstar
        if jstar >= 0:
            jstar = min(jstar, k)                              # can only be k if very late
            vec, mid = self._obs(self.tau(jstar), h_t + 1e-9)
            if mid is None:
                for jj in range(jstar - 1, -1, -1):
                    _, mid = self._obs(self.tau(jj), self.tau(jj + 1))
                    if mid is not None:
                        break
            self.her_last = mid if mid is not None else self.her_last
            p = model.init if jstar == 0 else model.predict(self.alpha[jstar - 1], self.pos[jstar], M)
            post = model.update(p, self._ll(vec, mid) if frac > 0 else np.zeros(N_CHORDS))
            for i in range(jstar + 1, k + 1):
                post = model.predict(post, self.pos[i], M)
            prior = post
        else:
            prior = model.init if k == 0 else _prior_chain(model, self.pos, M, k)
        rep = self._repeat_guess(k, jstar, when) if self.repeat else None
        if rep is not None:
            prior = model.update(prior, self._ll(*self._obs(self.tau(rep), self.tau(rep + 1))))
        horizon = [model.trans_key(self.pos[i], M) for i in range(k + 1, min(self.K, k + cfg.N))]
        z0, h0, path = self.planner.plan(prior, horizon, self.h_last, self.her_last)
        q = float(prior[z0])
        if not self.joined and k >= cfg.J and q >= cfg.q_min and self.q_prev >= cfg.q_min:
            self.joined = True
        drift = self._drift_ms() if self.monitor else None
        if self.joined and drift is not None and abs(drift) > cfg.drift_max:
            self.stopped = True
        action, h = 'wait', -1
        if self.stopped:
            action = 'stopped'
        elif self.joined:
            if q >= cfg.q_min:
                action, h = 'sing', h0
            elif self.h_last is not None and (self.h_last - self.tonic) % 12 in CT[z0]:
                action, h = 'hold', self.h_last
            else:
                action = 'silent'
            if h >= 0:
                self.h_last = h
        self.q_prev = q
        start = int(round((self.tau(k) - self.Lo) * SR))
        late = max(0, self.voice.pos - start)
        if late:
            self.late.append(late)
        if h >= 0 and not self.listen_only:
            dur = self.T_b - 0.03
            prev = self.last_note
            if prev and prev[2] == h and prev[3] == k - 1:     # same note on the next beat: hold it
                self.voice.set_dur(prev[0], (start - prev[1]) / SR + dur)
                self.last_note = (prev[0], prev[1], h, k)
            else:
                self.voice.schedule(start, dur, h, tag=k)
                self.last_note = (k, start, h, k)
        else:
            self.last_note = None
        b = {'k': k, 'pos': self.pos[k], 'heard_beats': round(when, 3), 'her': self.her_last, 'z_hat': z0, 'repeat_of': rep,
             'q': round(q, 3), 'action': action, 'h': h, 'plan': path, 'pi': [round(float(v), 4) for v in prior],
             'drift': None if drift is None else round(drift, 1), 'late_ms': round(late / SR * 1000, 1)}
        self.beats.append(b)
        bar, beat = divmod(k, M)
        self.log(f'{bar + 1:3d}.{beat + 1}  she {note_name(self.her_last):4s}  chord {chord_name(z0, self.tonic):9s} '
                 f'q {q:.2f}  Duet {action:7s} {note_name(h) if h >= 0 else "":4s}'
                 + (f'  repeat of {rep // M + 1}.{rep % M + 1}' if rep is not None else '')
                 + (f'  drift {drift:+.0f} ms' if drift is not None else '') + (f'  LATE {late / SR * 1000:.0f} ms' if late else ''))

    # ------------------------------------------------------------------ the repetition rule (exp_ceiling.Repeat, from audio)
    def _grid_update(self):
        """Her pitch on a fixed grid of 4 points per beat from beat 0 on (0 = rest), each point filled
        once the eighth of a beat around it has been heard."""
        while True:
            n = len(self.grid)
            x = self.t0 + (n / 4 + 1 / 8) * self.T_b
            if self.heard() < x + self.T_b / 8:
                return
            t = np.asarray(self.ft_t[-200:])
            sel = np.asarray(self.ft_v[-200:], bool) & (np.abs(t - x) <= self.T_b / 8)
            m = np.asarray(self.ft_m[-200:])[sel]
            self.grid.append(int(np.round(np.median(m))) if len(m) else 0)

    def _repeat_guess(self, k, jstar, when, C=2):
        """The most recent beat j (fully heard, same bar position) whose preceding C beats sound like
        the C beats she just sang; Duet then guesses she will sing beat j again. Returns j or None."""
        if jstar < C + 1:
            return None
        self._grid_update()
        g = np.asarray(self.grid)
        end = len(g)                                            # grid points heard so far
        ctx = g[end - 4 * C:end]
        if len(ctx) < 4 * C or (ctx > 0).mean() < 0.5:
            return None
        for j in range(jstar - 1, C - 1, -1):
            if self.pos[j] != self.pos[k]:
                continue
            s = end - 4 * (k - j)                               # the same moment, k - j beats earlier
            if s - 4 * C < 0:
                break
            old = g[s - 4 * C:s]
            both = (ctx > 0) | (old > 0)
            if both.sum() and (ctx[both] == old[both]).mean() >= 0.75:
                return j
        return None

    def summary(self):
        her = [(t, m) for t, m, v in zip(self.ft_t, self.ft_m, self.ft_v) if v]
        return {'cfg': asdict(self.cfg), 'tonic': self.tonic, 'M': self.M, 'K': self.K, 't0': self.t0, 'Li': self.Li,
                'Lo': self.Lo, 'late_decisions': len(self.late), 'voiced_frames': len(her), 'frames': len(self.ft_t),
                'proc_ms_median': float(np.median(self.proc_ms)) if self.proc_ms else None,
                'proc_ms_max': float(np.max(self.proc_ms)) if self.proc_ms else None,
                'stopped': self.stopped, 'beats': self.beats}


# ---------------------------------------------------------------------- the singer's pitch meter
def meter_line(engine):
    """One status line: the note she is singing now and how far off it is, in cents (100 = a semitone)."""
    if not engine.ft_t or not engine.ft_v[-1]:
        return '  pitch  --      ' + ' ' * 10 + '|' + ' ' * 10
    m = float(engine.ft_m[-1])
    near = int(round(m))
    cents = (m - near) * 100
    pos = int(round(np.clip(cents / 50, -1, 1) * 10))
    bar = [' '] * 21
    bar[10] = '|'
    bar[10 + pos] = '#'
    tag = 'in tune' if abs(cents) < 15 else ('sharp' if cents > 0 else 'flat')
    return f'  pitch  {note_name(near):4s} {cents:+4.0f} c  [{"".join(bar)}]  {tag}'


# ---------------------------------------------------------------------- the sound card
def run_live(engine, device=None, record=None, latency='low', meter=True):
    import sounddevice as sd
    rec_in, rec_out = [], []
    lock = threading.Lock()
    finished = threading.Event()
    say = engine.log
    engine.log = lambda s: print('\r\033[K' + s, flush=True) if meter else say(s)

    def cb(indata, outdata, frames, time_info, status):
        if status:
            engine.log(f'  [audio] {status}')
        x = indata[:, 0].copy()
        with lock:
            y = engine.process(x) if frames == HOP else np.zeros(frames)
        outdata[:, 0] = y
        outdata[:, 1] = y
        if record is not None:
            rec_in.append(x)
            rec_out.append(y.astype(np.float32))
        if engine.done():
            finished.set()

    stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=(1, 2), dtype='float32', latency=latency,
                       device=device, callback=cb)
    li, lo = stream.latency
    engine.Li, engine.Lo = li, lo
    engine.log(f'stream latency: in {li * 1000:.1f} ms, out {lo * 1000:.1f} ms;  '
               f'{engine.cfg.bpm:.0f} BPM, count-in {engine.count_in} clicks, then sing.')
    with stream:
        try:
            while not finished.wait(0.1):
                if meter:
                    with lock:
                        line = meter_line(engine)
                    print('\r\033[K' + line, end='', flush=True)
        except KeyboardInterrupt:
            engine.log('stopped by Ctrl-C')
    if meter:
        print('\r\033[K', end='')
    engine.log = say
    x_in = np.concatenate(rec_in) if rec_in else np.zeros(0)
    if record and rec_in:
        st = np.stack([x_in, np.concatenate(rec_out)], 1)
        write_wav(record, np.clip(st, -1, 1).astype(np.float32))
    return x_in


# ---------------------------------------------------------------------- static mode: sing first, Duet answers after
def static_take(mic, engine, model, cfg, out_prefix=None, play=False):
    """Harmonise a recorded take with everything heard (static.run_static); returns (beats, stereo mix)."""
    from audio import AudioSource, click, duet_notes, mix
    from static import run_static
    her = mic[int(round(engine.Li * SR)):].astype(float)          # mic counter -> acoustic time
    song = {'tonic': engine.tonic, 'M': engine.M, 'K': engine.K, 'sop': [],
            'beats': [{'pos': k % engine.M, 'z': -1} for k in range(engine.K)]}
    src = AudioSource(her, engine.t0, engine.T_b, engine.K)
    beats = run_static(song, model, replace(cfg, J=0), source=src)
    os.makedirs(OUT, exist_ok=True)
    n = len(her)
    clk = click(n, engine.t0, engine.T_b, engine.K, engine.M, engine.count_in)
    wav = (out_prefix + '_static.wav') if out_prefix else os.path.join(OUT, '_static.wav')
    mp3 = (out_prefix + '_static.mp3') if out_prefix else None
    st = mix(her.astype(np.float32), duet_notes(beats, engine.t0, engine.T_b), clk, wav, mp3)
    for bar in range(0, engine.K, engine.M):
        row = beats[bar:bar + engine.M]
        print(f'  bar {bar // engine.M + 1:2d}  ' + '  '.join(f'{chord_name(b.z_hat, engine.tonic):9s}{note_name(b.h):4s}' for b in row))
    if play:
        import sounddevice as sd
        print('playing her take with Duet ...')
        sd.play(st, SR)
        sd.wait()
    return beats, st, wav


# ---------------------------------------------------------------------- offline check: synthetic Twinkle
def selftest(model, cfg, style='clean', Li=0.012, Lo=0.020, repeat=True):
    """Feed a rendered Twinkle through LiveDuet block by block and score it like the batch path."""
    from audio import render_her
    from metrics import song_counts, summarize
    from planner import Beat
    from songs import twinkle
    song = twinkle()
    cfg = replace(cfg, bpm=song['bpm'])
    eng = LiveDuet(model, cfg, tonic=0, M=4, K=song['K'], count_in=4, Li=Li, Lo=Lo, repeat=repeat, log=lambda s: None)
    her = render_her(song, eng.t0, eng.T_b, style)               # her acoustic timeline
    shift = int(round(Li * SR))                                  # the mic hears it Li later
    x = np.concatenate([np.zeros(shift), her, np.zeros(SR)])
    n_blocks = int((eng.tau(song['K']) + 0.5 + Li) * SR) // HOP
    mic = []
    for b in range(n_blocks):
        blk = np.pad(x[b * HOP:(b + 1) * HOP], (0, max(0, HOP - len(x[b * HOP:(b + 1) * HOP]))))
        mic.append(blk)
        eng.process(blk)
    beats = [Beat(k=d['k'], pos=d['pos'], z_true=song['beats'][d['k']]['z'], z_hat=d['z_hat'], q=d['q'],
                  action=d['action'], h=d['h']) for d in eng.beats]
    sm = summarize([song_counts(song, beats, cfg.J)])
    return eng, sm, np.concatenate(mic), song


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bpm', type=float, default=80.0)
    ap.add_argument('--key', default='C', help='major key of the song (tonic), e.g. C, G, F, Bb')
    ap.add_argument('--meter', type=int, default=4)
    ap.add_argument('--beats', type=int, default=48, help='beats to sing after the count-in')
    ap.add_argument('--count-in', type=int, default=None, help='clicks before beat 1 (default: one bar)')
    ap.add_argument('--style', choices=('slow', 'bach'), default='bach', help="chord habits: 'slow' (pop/folk, chords mostly per bar) or 'bach'")
    ap.add_argument('--mode', choices=('long', 'vec'), default='long')
    ap.add_argument('--join', type=int, default=4, help='beats Duet listens before it may join')
    ap.add_argument('--range', type=int, nargs=2, default=list(ALTO), metavar=('LO', 'HI'),
                    help='MIDI range for Duet (default F3..F5; a low voice may want 41 65)')
    ap.add_argument('--listen', action='store_true', help='mic test: Duet decides but stays silent (click still plays)')
    ap.add_argument('--static', action='store_true', help='sing the whole take on the click first; Duet harmonises it afterwards and plays it back')
    ap.add_argument('--no-repeat', action='store_true', help='live: do not use the song\'s own repetitions')
    ap.add_argument('--no-meter', action='store_true', help='hide the pitch meter line')
    ap.add_argument('--no-monitor', action='store_true', help='do not stop when she drifts off the click')
    ap.add_argument('--device', default=None, help='sounddevice device (index or name); default: system default')
    ap.add_argument('--selftest', action='store_true', help='offline check with a synthetic singer, no sound card')
    a = ap.parse_args()

    res = json.load(open(os.path.join(HERE, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(a.range)
    cfg = replace(Config(**base), bpm=a.bpm, mode=a.mode, J=a.join)
    if a.mode == 'vec':
        cfg = replace(cfg, lam=res['tuning']['chosen']['vec']['lam'])
    model = load_model(a.style)

    if a.selftest:
        from metrics import song_counts, summarize
        for style in ('clean', 'vibrato'):
            for rep in (False, True):
                eng, sm, mic, song = selftest(model, replace(cfg, J=8), style, repeat=rep)
                s = eng.summary()
                fired = sum(b['repeat_of'] is not None for b in eng.beats)
                print(f"live   {style:8s} repeat {'on ' if rep else 'off'}  chords {sm['m1']:.3f}  in chord {sm['m2']:.3f}  "
                      f"consonant {sm['m3']:.3f}  repeat fired {fired:2d}  late {s['late_decisions']}  "
                      f"block {s['proc_ms_median']:.2f} / {s['proc_ms_max']:.2f} ms")
            beats, _, _ = static_take(mic, eng, model, replace(cfg, bpm=song['bpm']),
                                      out_prefix=os.path.join(OUT, f'selftest_{style}'))
            for b in beats:
                b.z_true = song['beats'][b.k]['z']
            sm = summarize([song_counts(song, beats, 8)])
            print(f"static {style:8s}               chords {sm['m1']:.3f}  in chord {sm['m2']:.3f}  consonant {sm['m3']:.3f}")
        return

    tonic = KEYS[a.key]
    eng = LiveDuet(model, cfg, tonic=tonic, M=a.meter, K=a.beats, count_in=a.count_in or a.meter,
                   listen_only=a.listen or a.static, monitor=not (a.no_monitor or a.static), repeat=not a.no_repeat)
    if a.static:
        eng.log = lambda s: None if s[:4].strip().replace('.', '').isdigit() else print(s)
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    prefix = os.path.join(OUT, stamp)
    print(f'key {a.key} major, {a.style} chord habits, Duet range {note_name(a.range[0])}..{note_name(a.range[1])}'
          + ('  [LISTEN ONLY]' if a.listen else '') + ('  [STATIC: sing first, Duet answers after]' if a.static else '')
          + '\nUse headphones. Ctrl-C stops.')
    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    mic = run_live(eng, device=dev, record=prefix + '.wav', meter=not a.no_meter)
    s = eng.summary()
    with open(prefix + '.json', 'w') as fh:
        json.dump(s, fh, default=float)
    sung = [b for b in s['beats'] if b['h'] >= 0]
    print(f"\nvoiced {s['voiced_frames']}/{s['frames']} frames, Duet sang {len(sung)}/{len(s['beats'])} beats, "
          f"late decisions {s['late_decisions']}, block {s['proc_ms_median']:.2f} ms median / {s['proc_ms_max']:.2f} max"
          + ('  (stopped: off the click)' if s['stopped'] else ''))
    print('log', prefix + '.json', '\nrecording (L mic, R click+Duet)', prefix + '.wav')
    if a.static and len(mic):
        _, _, wav = static_take(mic, eng, model, cfg, out_prefix=prefix, play=True)
        print('static mix', wav)


if __name__ == '__main__':
    main()
