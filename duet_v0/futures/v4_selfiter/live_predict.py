"""Duet with look-ahead: each beat's note is chosen BEFORE the beat and sent early by the output latency, so it
sounds on the beat together with her (README: follow, predict, send early), with a bench3 chord guesser inside.

PredictDuet = live_bench3.Bench3Duet (same tracker, same online notes, same reed voice) plus a beat plan:
- Beat 0 is her first note (heard, so that one note is answered late); from then on the grid is t0 + k * T_b.
- For each beat k, PRE + the output latency before the beat, Duet forecasts the chord of beat k from everything
  heard so far: the system's HMM filter run over the beats already heard and pushed one beat on with its
  transition table (her sounding note counts W_CUR as evidence for beat k: she may hold it), mixed the way the
  system mixes (r2_mix_v2: HMM^0.3 x GRU^0.7, the GRU's last belief pushed on by the same table when her last
  note started before beat k). It then sings the chord tone m1's chooser likes most without knowing her next note:
  P(chord tone) minus m1's step / leap / middle penalties, plus W_KEEP for keeping the sounding note, below the
  lowest of her last notes. It plans only while she is singing (voiced within the last beat).
- When her note is heard (tracker, about 40-70 ms in), the system's own step() runs as in bench3 and gives the
  reactive choice. If the planned note is not consonant with what she sang, is her unison/octave, or is not below
  her, Duet switches to the reactive choice at once (a late correction); otherwise the planned note stays.
Systems without an m1-style chord filter (third_below, the style-aware HMM) stay reactive.

  .venv/bin/python futures/v4_selfiter/live_predict.py --evaluate     # offline: phone files, reactive vs predictive
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_bench3 as LB  # noqa: E402  (puts duet_beat, live, panel and v3 on sys.path)
from live_bench3 import HOP, SR, Bench3Duet  # noqa: E402
from systems import m1_filter_v5lo as M1  # noqa: E402
from theory import ALTO, ALTO_MID, CONSONANT  # noqa: E402


class PredictDuet(Bench3Duet):
    PRE = 0.025            # s: start this much before the beat (the reed takes 60 ms to reach full level)
    W_KEEP = 0.15          # bonus for keeping the sounding note on the next beat
    W_CUR = 0.5            # weight of her sounding note as evidence for the next beat

    def __init__(self, system, name, *a, count_in=0, click_all=False, lead=0.5, **kw):
        super().__init__(system, name, *a, **kw)
        self.can_plan = isinstance(system, M1.Filter)
        self.next_k = 1
        self.h = -1                                          # the note Duet is sounding (or about to)
        self.n_tag = 0
        self.plans = []
        self.count_in, self.click_all = count_in, click_all
        if count_in:                                         # she sings on Duet's click: the grid is known ahead
            self.t0 = lead + count_in * self.T_b
            self.next_k = 0

    def _click(self, n):
        """LiveDuet's click (count-in, or every beat with click_all), sent early by the output latency."""
        if not self.count_in:
            return np.zeros(n)
        y = np.zeros(n)
        last = self.K if self.click_all else 0
        for k in range(-self.count_in, last):
            i0 = int(round((self.t0 + k * self.T_b - self.Lo) * SR)) - self.n
            L = int(0.025 * SR)
            if i0 + L <= 0 or i0 >= n:
                continue
            tt = np.arange(L) / SR
            f = 1760.0 if k % self.M == 0 else 1320.0
            s = (0.35 if k % self.M == 0 else 0.22) * np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.006)
            a, b = max(i0, 0), min(i0 + L, n)
            y[a:b] += s[a - i0:b - i0]
        return y

    # ---------------------------------------------------------------- one block
    def process(self, x):
        import time
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
        self._plan()
        y = self.voice.render(HOP) + self._click(HOP) + self.monitor * x
        self.n += HOP
        self.proc_ms.append((time.perf_counter() - c0) * 1000)
        return y

    def _sing(self, h, start):
        """From output sample `start` on, Duet sings h (-1: nothing); the sounding note ends there."""
        if self.tag is not None:
            nt = self.voice.by_tag.get(self.tag)
            if nt is not None:
                self.voice.set_dur(self.tag, max(0.0, (start - nt.i0) / SR))
            self.tag = None
        if h >= 0 and not self.listen_only:
            self.n_tag += 1
            self.tag = ('p', self.n_tag)
            self.voice.schedule(start, 60.0, h, tag=self.tag)
        self.h = h

    # ---------------------------------------------------------------- her note heard: keep or correct
    def _react(self, t):
        c = self.cur
        need = 12 if self.t0 is None else 3
        if c is not None and sum(c[2].values()) >= need and c[0] != self.answered and not self.ended:
            self.answered = c[0]
            midi = int(max(c[2], key=c[2].get))
            first = self.t0 is None
            if first:
                self.t0 = c[0] - self.first_beat * self.T_b
                self.next_k = int(np.floor(self.first_beat)) + 1
            onset_raw = (c[0] - self.t0) / self.T_b
            onset = self._snap(onset_raw)
            if onset >= self.K:
                self.ended = True
            else:
                prev = self.notes[-1] if self.notes else None
                prev_dur = None if not self.events or prev is None else self._snap((prev[1] - prev[0]) / self.T_b, True)
                i = len(self.events)
                h_r = int(self.system.step(i, float(onset), midi, prev_dur))
                iv = abs(midi - self.h) % 12 if self.h >= 0 else None
                if self.KEEP == 'same':                          # the plan stays only if the reactive choice agrees
                    keep = self.can_plan and not first and self.h >= 0 and h_r == self.h
                else:                                            # 'consonant': any consonant note below her stays
                    keep = (self.can_plan and not first and self.h >= 0 and self.h < midi and iv in CONSONANT and iv != 0)
                if keep:
                    action = 'kept'
                    if hasattr(self.system, 'prev'):
                        self.system.prev = self.h                # m1's voice leading follows what Duet sings
                else:
                    action = 'first' if first else ('late' if not self.can_plan or self.h < 0 else 'corrected')
                    if h_r != self.h:
                        self._sing(h_r, self.voice.pos)
                lag = (t - c[0]) * 1000
                self.events.append({'i': i, 'onset_beats': round(onset, 3), 'onset_raw': round(onset_raw, 3),
                                    'her_heard': midi, 'duet': self.h, 'reactive': h_r, 'action': action,
                                    'prev_dur': None if prev_dur is None else round(prev_dur, 3),
                                    'detect_ms': round(lag, 1)})
                bar, beat = divmod(int(np.floor(onset + 1e-6)), self.M)
                self.log(f'{bar + 1:3d}.{beat + 1}  she {LB.lm.note_name(midi):4s} -> Duet '
                         f'{LB.lm.note_name(self.h) if self.h >= 0 else "silent":6s}  {action}')
        if self.tag is not None and (self.cur is None and self.last_voiced is not None
                                     and t - self.last_voiced > self.hold_gap):
            self._sing(-1, self.voice.pos)                       # she stopped: Duet stops too
        if self.ended and self.tag is not None:
            self._sing(-1, self.voice.pos)

    # ---------------------------------------------------------------- the beat plan
    def _plan(self):
        if not self.can_plan or self.t0 is None or self.ended:
            return
        while self.next_k < self.K:
            k = self.next_k
            start = int(round((self.t0 + k * self.T_b - self.Lo - self.PRE) * SR))
            if start >= self.n + 2 * HOP:
                return                                          # not yet: decide as late as possible
            self.next_k += 1
            singing = self.last_voiced is not None and self.heard() - self.last_voiced < self.T_b
            if not singing:
                self.plans.append({'k': k, 'h': None, 'why': 'silent'})
                continue
            b = self._forecast(k)
            h = self._choose(b)
            late = max(0, self.voice.pos - start)
            self.plans.append({'k': k, 'h': h, 'top': int(np.argmax(b)), 'p': round(float(b.max()), 3),
                               'late_ms': round(late / SR * 1000, 1)})
            if late:
                self.late.append(late)
            if h != self.h:
                self._sing(h, max(start, self.voice.pos))

    def _forecast(self, k):
        f = self.system
        a = f.alpha
        for j in range(f.done, k):
            a = M1.Filter._advance(f, a, j, f._beat_ll(j))
        ll = self.W_CUR * f._beat_ll(k) if self.W_CUR else np.zeros(M1.S)
        bh = M1.Filter._advance(f, a, k, ll)
        p = getattr(f, '_p', None)
        if p is None:
            return bh
        bg = np.asarray(p, dtype=np.float64)
        if f.notes and int(np.floor(f.notes[-1][0] + 1e-9)) < k:
            bg = bg @ f.m.A[M1.tkey(f._pos(k), f.M)]
        al = getattr(f, 'a', 0.7)
        lg = (1 - al) * np.log(np.maximum(bh, 1e-300)) + al * np.log(np.maximum(bg, 1e-300))
        w = np.exp(lg - lg.max())
        return w / w.sum()

    R_REPEAT = 0.3         # in 'expect' choice: chance her next note repeats the sounding one
    CHOICE = 'mimic'       # 'mimic': m1's chooser as if she repeats her note; 'tone': most likely chord tone;
                           # 'expect': expected music_nd over her likely next pc
    W_NOW = 0.0            # bonus for a note consonant with her sounding note (not her unison/octave)
    # Chosen on Twinkle (phone file, 41 notes, --evaluate, 2026-11-01): 'mimic' + 'consonant' for r2_mix_v2
    # (fit while she sings 0.665 vs reactive 0.638; 19/41 notes on time). One song only: not a val-set result.
    KEEP = 'consonant'     # 'same': keep the plan only if the reactive choice agrees; 'consonant': if consonant
    _CONS = np.array([1.0 if i in (3, 4, 7, 8, 9) else 0.0 for i in range(12)])   # her pc - Duet pc, Duet below

    def _mimic(self, b):
        """m1's chooser on the forecast belief, with her sounding note standing in for her next one."""
        f = self.system
        her = [int(n[2]) for n in self.notes[-1:]]
        if self.cur is not None:
            her = [int(max(self.cur[2], key=self.cur[2].get))]
        if not her:
            return -1
        midi = her[-1]
        pfit = b @ M1.CTM
        ref = self.h if self.h >= 0 else (f.prev if getattr(f, 'prev', None) is not None else ALTO_MID)
        best, bs = -1, -1e9
        for h in range(ALTO[0], min(ALTO[1], midi - 1) + 1):
            iv = abs(midi - h) % 12
            if iv not in CONSONANT:
                continue
            s = pfit[(h - f.t) % 12]
            if iv == 0:
                s -= M1.P['w_oct']
            d = abs(h - ref)
            s -= M1.P['w_step'] * d + M1.P['w_big'] * (d > 7) + M1.P['w_mid'] * abs(h - ALTO_MID)
            if s > bs:
                best, bs = h, s
        return best

    def _choose(self, b):
        if self.CHOICE == 'mimic':
            return self._mimic(b)
        f = self.system
        pfit = b @ M1.CTM
        if self.CHOICE == 'expect':
            q = f.m.Bc / f.m.Bc.sum(1, keepdims=True)                        # P(her pc | chord), relative pcs
            if self.cur is not None and self.R_REPEAT:
                rep = np.zeros(12)
                rep[(int(max(self.cur[2], key=self.cur[2].get)) - f.t) % 12] = 1.0
                q = (1 - self.R_REPEAT) * q + self.R_REPEAT * rep[None, :]
            # E[fit](pc) = sum_z b(z) CT(z, pc) sum_p q_z(p) cons(p - pc)
            cons = np.array([[self._CONS[(p - c) % 12] for c in range(12)] for p in range(12)])   # [p, c]
            pfit = (b[:, None] * M1.CTM * (q @ cons)).sum(0)
        her = [int(n[2]) for n in self.notes[-3:]]
        if self.cur is not None:
            her.append(int(max(self.cur[2], key=self.cur[2].get)))
        hi = min(ALTO[1], min(her) - 1) if her else ALTO[1]
        ref = self.h if self.h >= 0 else (f.prev if getattr(f, 'prev', None) is not None else ALTO_MID)
        now = her[-1] if her else None                       # her sounding (or last) note: she often repeats it
        best, bs = -1, -1e9
        for h in range(ALTO[0], hi + 1):
            d = abs(h - ref)
            s = pfit[(h - f.t) % 12] - M1.P['w_step'] * d - M1.P['w_big'] * (d > 7) - M1.P['w_mid'] * abs(h - ALTO_MID)
            if self.W_NOW and now is not None:
                iv = (now - h) % 12
                s += self.W_NOW * ((iv in CONSONANT and iv != 0) - (iv == 0))
            if h == self.h:
                s += self.W_KEEP
            if s > bs:
                best, bs = h, s
        return best

    def done(self):
        return self.ended or (self.t0 is not None and self.heard() > self.t0 + (self.K + 1) * self.T_b)

    def summary(self):
        s = super().summary()
        acts = [e.get('action') for e in self.events]
        s.update({'mode': 'predict' if self.can_plan else 'reactive', 'kept': acts.count('kept'),
                  'corrected': acts.count('corrected'), 'late_start': acts.count('late'), 'plans': self.plans})
        return s


# ---------------------------------------------------------------------- offline evaluation
def sounding(eng, n_blocks_before):
    """Midi Duet sounds at the start of the block about to be rendered (-1: none)."""
    s0 = eng.voice.pos
    h = -1
    for nt in eng.voice.notes:
        if nt.i0 <= s0 < nt.i0 + int(nt.dur * SR):
            h = int(nt.midi)
    return h


def run_file(eng, x, Lo=0.010):
    eng.Lo = Lo
    track = []
    for b in range(len(x) // HOP):
        eng.process(x[b * HOP:(b + 1) * HOP])
        track.append(sounding(eng, b))                   # sounds at (b + 1) * HOP / SR + Lo acoustically
    return np.array(track)


def score_twinkle(track, Lo=0.010, beat1=4.0, T_b=0.75, J=4):
    """Time share of her singing (from beat J) where Duet's SOUNDING note is consonant with her true note, a tone
    of the true chord and not her unison/octave, counting every millisecond Duet is late as a miss."""
    from songs import twinkle
    from theory import CT
    song = twinkle()
    z = [b['z'] for b in song['beats']]
    t_out = (np.arange(len(track)) + 2) * HOP / SR + Lo        # when each tracked block is heard
    tot = good = late_wrong = first100 = 0
    for on, off, m in sorted(song['sop']):
        a, e = beat1 + on * T_b, beat1 + off * T_b - 0.03
        for idx in np.where((t_out >= a) & (t_out < e))[0]:
            k = int(np.floor((t_out[idx] - beat1) / T_b))
            if k < J or k >= len(z):
                continue
            h = track[idx]
            ok = h >= 0 and (abs(m - h) % 12) in CONSONANT and (abs(m - h) % 12) != 0 and (h % 12) in CT[z[k]]
            tot += 1
            good += ok
            if t_out[idx] - a < 0.10:
                first100 += 1
                late_wrong += not ok
    return good / tot, late_wrong / max(first100, 1)


def onset_lag(eng, track, x_true_onsets, Lo=0.010):
    """For each true onset (s), ms until Duet sounds the note it will keep for that onset."""
    t_out = (np.arange(len(track)) + 2) * HOP / SR + Lo
    lags = []
    for a in x_true_onsets:
        win = (t_out >= a - 0.2) & (t_out < a + 0.4)
        idx = np.where(win)[0]
        if not len(idx):
            continue
        final = track[idx[-1] if t_out[idx[-1]] < a + 0.35 else idx[np.searchsorted(t_out[idx], a + 0.3)]]
        sel = idx[(track[idx] == final)]
        if final < 0 or not len(sel):
            continue
        first = sel[0]
        lags.append((t_out[first] - a) * 1000)
    return np.array(lags)


def evaluate():
    import soundfile as sf
    import bench3
    from songs import twinkle
    train = bench3.train_sets()
    x, sr = sf.read(os.path.join(LB.ABS, 'futures', 'live_test', 'phone', 'twinkle_C_80bpm_countin.wav'),
                    dtype='float32', always_2d=True)
    x = x.mean(1)
    song = twinkle()
    onsets = [4.0 + on * 0.75 for on, _, _ in sorted(song['sop'])][1:]
    for name in ('r2_mix_v2', 'm1_filter_v5lo'):
        sysm = LB.build(name, train)
        variants = [('reactive', None, None, None)] + [('predict', c, k, wc) for c in ('tone', 'mimic')
                                                         for k in ('consonant', 'same') for wc in (0.0, 0.5)]
        for mode, choice, keep, wc in variants:
            if mode == 'reactive':
                eng = Bench3Duet(sysm, name, 0, 4, 48, 80.0, log=lambda s: None)
            else:
                eng = PredictDuet(sysm, name, 0, 4, 48, 80.0, log=lambda s: None)
                eng.CHOICE, eng.KEEP, eng.W_CUR, eng.W_NOW = choice, keep, wc, 0.0
            tr = run_file(eng, x)
            nd, lw = score_twinkle(tr)
            s = eng.summary()
            extra = '' if mode == 'reactive' else f"  on time {s['kept']:2d}/41 switched {s['corrected']:2d}"
            print(f'{name:15s} {mode:8s} {choice!s:5s} keep {keep!s:9s} W_CUR {wc!s:4s}  fit while she sings {nd:.3f}  '
                  f'wrong in her first 100 ms {lw:.2f}{extra}', flush=True)


if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        evaluate()
