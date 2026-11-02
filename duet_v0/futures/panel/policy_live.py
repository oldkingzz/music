"""The leaderboard's best system (bench_systems/policy.py, 0.793) inside the live engine (duet_beat/live_mic.py).

PolicyDuet is LiveDuet with one change: the note for each beat comes from the policy instead of the chord
HMM + planner. Everything else is LiveDuet's own: the 10 ms feature tracker, the click, the acoustic
timeline (Li, Lo), scheduling each note early by the output latency, holding a repeated note.
Her melody is turned into notes online from the tracker's frames (a note starts at a detected onset or
when the rounded pitch has changed for 30 ms, and ends at silence), in beat units from beat 0. At the
decision for beat k the policy sees exactly what has been heard: notes up to the newest frame, the
sounding one clipped there (SongState with D = k - heard beats). Duet joins after cfg.J beats.
"""
import math

import numpy as np

import live_mic as lm
from bench_systems import policy as pol
from theory import ALTO_MID

NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
MAJOR = (0, 2, 4, 5, 7, 9, 11)
DIATONIC = True                 # Duet sings only notes of the song's major scale (the bench policy may pick any pc)
_SYS = {}


def load_policy():
    """The trained leaderboard policy (weights under bench/models/policy; never retrains here)."""
    if 'sys' not in _SYS:
        import os
        import bench
        train = bench.train_sets()
        fp, _ = pol._fingerprint(train)
        if not os.path.exists(os.path.join(pol.MDIR, f'policy_{fp}.npz')):
            raise RuntimeError(f'policy weights policy_{fp}.npz not found; refusing to retrain from the panel')
        _SYS['sys'] = pol.build(train)
    return _SYS['sys']


class PolicyDuet(lm.LiveDuet):
    def __init__(self, policy, *a, **kw):
        super().__init__(*a, **kw)
        self.policy = policy
        self.seg_i = 0                  # frames already segmented
        self.notes = []                 # closed notes: [on_s, off_s, midi]
        self.cur = None                 # [on_s, last_s, {midi: frames}, cand_midi, cand_start_s, cand_n]
        self.gap = 0
        self.guess = None               # policy's guess of her pitch class for the beat just decided

    # ---------------------------------------------------------------- frames -> notes (online)
    def _close(self):
        c = self.cur
        if c is not None:
            midi = max(c[2], key=c[2].get)
            if c[1] - c[0] >= 0.05:
                self.notes.append([c[0], c[1] + 0.01, midi])
        self.cur = None

    def _segment(self):
        on_set = set(np.round(np.asarray(self.on_t[-50:]), 4)) if self.on_t else set()
        while self.seg_i < len(self.ft_t):
            i = self.seg_i
            t, m, v = self.ft_t[i], self.ft_m[i], self.ft_v[i]
            self.seg_i += 1
            if not v:
                self.gap += 1
                if self.cur is not None and self.gap > 3:   # bridge up to 30 ms of unvoiced frames
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
                if c[5] >= 3:                                  # 30 ms on a new pitch: new note from its start
                    start = c[4]
                    c[1] = start - 0.01
                    self._close()
                    self.cur = [start, t, {q: 3}, None, t, 0]
                    continue
            else:
                c[3], c[5] = None, 0
            c[1] = t
            c[2][q] = c[2].get(q, 0) + 1

    def her_notes_beats(self, h_t):
        """Her notes heard so far, in beats from beat 0; the sounding one ends at the newest frame."""
        b = lambda s: (s - self.t0) / self.T_b  # noqa: E731
        out = [[b(on), b(off), m] for on, off, m in self.notes]
        if self.cur is not None:
            c = self.cur
            out.append([b(c[0]), b(min(c[1] + 0.01, h_t)), max(c[2], key=c[2].get)])
        return [n for n in out if n[1] > n[0]]

    # ---------------------------------------------------------------- one beat
    def _decide(self, k):
        cfg = self.cfg
        h_t = self.heard()
        self._segment()
        when = (h_t - self.t0) / self.T_b
        sop = self.her_notes_beats(h_t)
        st = pol.SongState(sop, self.tonic, self.pos, self.M, k - when, int(pol.P['ctx']))
        x = st.features(k)
        q, val = self.policy.forward(x)
        self.guess = int(np.argmax(q))
        her = st.her_last
        self.her_last = her if her is not None else self.her_last
        lo, hi = cfg.alto
        H = np.arange(lo, hi + 1)
        hpc = (H - self.tonic) % 12
        h0 = -1
        if self.her_last is not None:
            sc = val[hpc] - pol.P['wd'] * q[hpc] - pol.P['vc'] * np.abs(H - ALTO_MID)
            if self.h_last is not None:
                sc = sc - pol.P['vl'] * np.abs(H - self.h_last)
            ok = H <= self.her_last - 1
            if DIATONIC:
                ok &= np.isin(hpc, MAJOR)
            if ok.any():
                sc = np.where(ok, sc, -np.inf)
            h0 = int(H[int(np.argmax(sc))])
        if not self.joined and k >= cfg.J and self.her_last is not None:
            self.joined = True
        action, h = 'wait', -1
        if self.joined and h0 >= 0:
            action, h = 'sing', h0
            self.h_last = h
        start = int(round((self.tau(k) - self.Lo) * lm.SR))
        late = max(0, self.voice.pos - start)
        if late:
            self.late.append(late)
        if h >= 0 and not self.listen_only:
            dur = self.T_b - 0.03
            prev = self.last_note
            if prev and prev[2] == h and prev[3] == k - 1:     # same note on the next beat: hold it
                self.voice.set_dur(prev[0], (start - prev[1]) / lm.SR + dur)
                self.last_note = (prev[0], prev[1], h, k)
            else:
                self.voice.schedule(start, dur, h, tag=k)
                self.last_note = (k, start, h, k)
        else:
            self.last_note = None
        v = float(val[(h - self.tonic) % 12]) if h >= 0 else 0.0
        self.beats.append({'k': k, 'pos': self.pos[k], 'heard_beats': round(when, 3), 'her': self.her_last, 'z_hat': -1,
                           'repeat_of': None, 'q': round(v, 3), 'action': action, 'h': h, 'plan': [],
                           'pi': [], 'drift': None, 'late_ms': round(late / lm.SR * 1000, 1),
                           'guess_pc': self.guess, 'notes_heard': len(sop)})
        bar, beat = divmod(k, self.M)
        self.log(f'{bar + 1:3d}.{beat + 1}  she {lm.note_name(self.her_last):4s}  guess {NAMES[(self.tonic + self.guess) % 12]:2s}  '
                 f'Duet {action:5s} {lm.note_name(h) if h >= 0 else "":4s}' + (f'  LATE {late / lm.SR * 1000:.0f} ms' if late else ''))


def selftest(style='vibrato', Li=0.012, Lo=0.020):
    """Rendered Twinkle through PolicyDuet block by block, scored like live_mic.selftest (same J, Li, Lo)."""
    import json
    import os
    from dataclasses import replace
    from audio import render_her
    from metrics import song_counts, summarize
    from planner import Beat, Config
    from songs import twinkle
    here = os.path.dirname(os.path.abspath(lm.__file__))
    res = json.load(open(os.path.join(here, 'out', 'results.json')))
    base = dict(res['tuning']['chosen']['main'])
    base['alto'] = tuple(base['alto'])
    song = twinkle()
    cfg = replace(Config(**base), bpm=song['bpm'], J=4)
    out = {}
    for name in ('policy', 'hmm'):
        if name == 'policy':
            eng = PolicyDuet(load_policy(), lm.load_model('bach'), cfg, tonic=0, M=4, K=song['K'], count_in=4,
                             Li=Li, Lo=Lo, monitor=False, log=lambda s: None)
        else:
            eng = lm.LiveDuet(lm.load_model('bach'), cfg, tonic=0, M=4, K=song['K'], count_in=4, Li=Li, Lo=Lo,
                              monitor=False, log=lambda s: None)
        her = render_her(song, eng.t0, eng.T_b, style)
        x = np.concatenate([np.zeros(int(round(Li * lm.SR))), her, np.zeros(lm.SR)])
        nb = int((eng.tau(song['K']) + 0.5 + Li) * lm.SR) // lm.HOP
        for b in range(nb):
            blk = x[b * lm.HOP:(b + 1) * lm.HOP]
            eng.process(np.pad(blk, (0, lm.HOP - len(blk))))
        bts = [Beat(k=d['k'], pos=d['pos'], z_true=song['beats'][d['k']]['z'], z_hat=d['z_hat'], q=d['q'],
                    action=d['action'], h=d['h']) for d in eng.beats]
        sm = summarize([song_counts(song, bts, cfg.J)])
        s = eng.summary()
        out[name] = {'consonant': round(float(sm['m3']), 3), 'late': s['late_decisions'],
                     'block_ms_median': round(float(s['proc_ms_median']), 2), 'block_ms_max': round(float(s['proc_ms_max']), 2)}
    return out
