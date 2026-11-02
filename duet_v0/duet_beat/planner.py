"""Duet on a fixed beat grid: decide, before every beat, which note to sing (spec E5, E11-E15, E18-E21).

One loop serves every system so they differ only in where the chord belief comes from:
  hmm     the chord HMM, causal: at the decision time for beat k it has heard everything up to
          tau_k - L - delta (E5), so the beat before is usually cut short;
  prior   B1: the same model but it never listens (transitions only);
  tonic   B0: always chord I;
  peek    B3: cheats by seeing the whole of beat k's melody (not causal: an upper bound);
  oracle  B4: is told Bach's chord (upper bound for the note choice alone);
  rule    B2: the repository's rule, a diatonic third/sixth below her latest note, no chords.
The note is chosen by a joint plan over (chord, note) for N beats (E20) and only the first beat
is executed; the plan is redone every beat (E21, receding horizon).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from model import ChordHMM, soprano_obs
from theory import ALTO, ALTO_MID, CT, MAJOR, N_CHORDS, RULE_STEPS

EPS = 1e-12


@dataclass
class Config:
    mode: str = 'long'          # observation: 'long' (longest pitch class) or 'vec' (12-vector), Q3
    N: int = 4                  # planning horizon in beats (E19-E20)
    gamma: float = 0.1          # weight of voice leading against chord log-probability (E20)
    w1: float = 1.0             # cost per semitone of Duet's leap (E14)
    w3: float = 0.2             # cost per semitone away from the middle of the alto range (E14)
    q_min: float = 0.5          # confidence needed to sing a new note (E15, E18)
    J: int = 8                  # beats to listen before joining (E18)
    lam: float = 1.0            # evidence weight of the 12-vector observation (E8)
    bpm: float = 100.0          # beats per minute of the click (A2)
    L: float = 0.052            # output latency, s (A6)
    delta: float = 0.020        # compute margin, s (E5)
    alto: tuple = ALTO
    drift_max: float = 80.0     # E17: stop if her onsets sit this far (ms, median of 4) off the grid


@dataclass
class Beat:
    k: int
    pos: int
    z_true: int
    z_hat: int = -1             # planned chord for this beat (-1: none, e.g. the rule)
    q: float = 0.0              # belief in that chord at decision time
    action: str = 'wait'        # wait | sing | hold | silent | stopped (E17: she left the click)
    drift: float = float('nan') # E17 reading at decision time, ms
    h: int = -1                 # Duet's MIDI note (-1: silent)
    pi: list = field(default_factory=list)
    plan: list = field(default_factory=list)   # planned chords over the horizon


# ---------------------------------------------------------------------- observations
class ScoreSource:
    """Her melody from the score (symbolic simulation). Audio sources implement the same two calls."""

    def __init__(self, song):
        self.sop = song['sop']

    def full(self, k):
        return soprano_obs(self.sop, k)

    def partial(self, k, end_frac):
        return soprano_obs(self.sop, k, end_frac)


def to_obs(vec_abs, longest_midi, tonic, mode):
    if mode == 'long':
        return None if longest_midi is None else (longest_midi - tonic) % 12
    v = np.roll(np.asarray(vec_abs, float), -tonic)
    return v if v.sum() > 0 else None


# ---------------------------------------------------------------------- planner (E13, E14, E20)
class Planner:
    def __init__(self, tonic, cfg, model):
        self.model = model
        lo, hi = cfg.alto
        st = [(z, h) for z in range(N_CHORDS) for h in range(lo, hi + 1) if (h - tonic) % 12 in CT[z]]
        self.Z = np.array([s[0] for s in st])
        self.H = np.array([s[1] for s in st], float)
        self.cfg = cfg
        self.pitch_step = cfg.w1 * np.abs(self.H[:, None] - self.H[None, :]) + cfg.w3 * np.abs(self.H[None, :] - ALTO_MID)
        self._trans_cache = {}

    def _trans_cost(self, key):
        if key not in self._trans_cache:
            A = self.model.mat(key)
            self._trans_cache[key] = -np.log(A[self.Z[:, None], self.Z[None, :]] + EPS)
        return self._trans_cache[key]

    def mask(self, her_pitch):
        """E13: chord tones in the alto range, below her latest note (relaxed per chord if none fit)."""
        if her_pitch is None:
            return np.ones(len(self.H), bool)
        ok = self.H <= her_pitch - 1
        for z in range(N_CHORDS):
            sel = self.Z == z
            if not ok[sel].any():
                ok[sel] = True
        return ok

    def plan(self, prior, trans_keys, h_prev, her_pitch):
        """Joint (chord, note) path over len(trans_keys)+1 beats; returns the first beat's (z, h, path)."""
        cfg = self.cfg
        ok = self.mask(her_pitch)
        lead = cfg.w1 * np.abs(self.H - h_prev) if h_prev is not None else 0.0
        cost = -np.log(prior[self.Z] + EPS) + cfg.gamma * (lead + cfg.w3 * np.abs(self.H - ALTO_MID))
        cost = np.where(ok, cost, np.inf)
        backs = []
        for key in trans_keys:
            step = self._trans_cost(key) + cfg.gamma * self.pitch_step
            tot = cost[:, None] + step
            b = np.argmin(tot, axis=0)
            cost = np.where(ok, tot[b, np.arange(len(b))], np.inf)
            backs.append(b)
        s = int(np.argmin(cost))
        path = [s]
        for b in reversed(backs):
            s = int(b[s])
            path.append(s)
        path.reverse()
        return int(self.Z[path[0]]), int(self.H[path[0]]), [int(self.Z[p]) for p in path]


# ---------------------------------------------------------------------- the online loop
def run(song, model: ChordHMM, cfg: Config, system='hmm', source=None, hint=None):
    """Simulate one song. song: {'tonic','M','K','beats':[{'pos','z'}],'sop'} (beats in beat units).

    hint(k, jstar, frac), optional: a guess of what she will sing in beat k, made only from what was
    heard by the decision time; the 'hmm' belief is updated with it as if it had been heard."""
    source = source or ScoreSource(song)
    t, M, K = song['tonic'], song['M'], song['K']
    pos = [b['pos'] for b in song['beats']]
    ztrue = [b['z'] for b in song['beats']]
    D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)          # decision lead in beats (E5)
    planner = Planner(t, cfg, model)
    listen = system in ('hmm', 'peek')
    mode = cfg.mode

    def ll(vec, mid):
        if not listen:
            return np.zeros(N_CHORDS)
        return model.loglik(to_obs(vec, mid, t, mode), mode, cfg.lam)

    # full-beat forward filter: pi_full[j] = belief for beat j before hearing it, alpha_full[j] after
    pi_full, alpha_full = [], []
    for j in range(K):
        p = model.init if j == 0 else model.predict(alpha_full[j - 1], pos[j], M)
        pi_full.append(p)
        alpha_full.append(model.update(p, ll(*source.full(j))))

    out, joined, q_prev, h_last, her_last, stopped = [], False, 0.0, None, None, False
    monitor = getattr(source, 'drift_ms', None)
    for k in range(K):
        when = k - D                                    # time of the decision, in beats
        jstar = math.floor(when)                         # the beat being heard at that moment
        frac = when - jstar
        # her latest pitch we can know (for "stay below her", E13, and for the rule)
        if jstar >= 0:
            vec, mid = source.partial(jstar, frac)
            if mid is None and frac < 1:
                for jj in range(jstar - 1, -1, -1):
                    _, mid = source.full(jj)
                    if mid is not None:
                        break
            her_last = mid if mid is not None else her_last
        # chord belief for beat k at decision time
        if system == 'tonic':
            prior = np.eye(N_CHORDS)[0]
        elif system == 'oracle':
            prior = np.eye(N_CHORDS)[ztrue[k] if ztrue[k] >= 0 else 0]
        elif system == 'peek':
            prior = alpha_full[k]
        elif system in ('hmm', 'prior'):
            if jstar < 0:
                prior = pi_full[k] if system == 'hmm' else _prior_chain(model, pos, M, k)
            else:
                if system == 'hmm':
                    post = model.update(pi_full[jstar], ll(*source.partial(jstar, frac)))
                else:
                    post = _prior_chain(model, pos, M, jstar)
                for i in range(jstar + 1, k + 1):
                    post = model.predict(post, pos[i], M)
                prior = post
            if hint is not None and system == 'hmm':
                guess = hint(k, jstar, frac)                # a guess of beat k's melody: (vec, longest_midi) or None
                if guess is not None:
                    prior = model.update(prior, ll(*guess))
        else:
            prior = None

        bt = Beat(k=k, pos=pos[k], z_true=ztrue[k])
        if system == 'rule':
            if k >= cfg.J and her_last is not None:
                h = next((her_last - iv for iv in RULE_STEPS if (her_last - iv - t) % 12 in MAJOR), her_last - 3)
                bt.action, bt.h, bt.q = 'sing', int(h), 1.0
                h_last = bt.h
            out.append(bt)
            continue

        horizon = [model.trans_key(pos[i], M) for i in range(k + 1, min(K, k + cfg.N))]
        if system in ('tonic', 'oracle'):
            horizon = []                                 # chord is given: nothing to plan ahead
        z0, h0, path = planner.plan(prior, horizon, h_last, her_last)
        q = 1.0 if system in ('tonic', 'oracle') else float(prior[z0])
        bt.z_hat, bt.q, bt.pi, bt.plan = z0, q, [round(float(x), 4) for x in prior], path
        if not joined and k >= cfg.J and q >= cfg.q_min and q_prev >= cfg.q_min:
            joined = True
        if monitor is not None:
            dr = monitor(source.t0 + when * source.T_b)
            if dr is not None:
                bt.drift = round(dr, 1)
                if joined and abs(dr) > cfg.drift_max:
                    stopped = True                       # E17: stop and raise the alarm (sticky)
        if stopped:
            bt.action = 'stopped'
        elif joined:
            if q >= cfg.q_min:
                bt.action, bt.h = 'sing', h0
            elif h_last is not None and (h_last - t) % 12 in CT[z0]:
                bt.action, bt.h = 'hold', h_last
            else:
                bt.action = 'silent'
            if bt.h >= 0:
                h_last = bt.h
        q_prev = q
        out.append(bt)
    return out


def _prior_chain(model, pos, M, j):
    p = model.init
    for i in range(1, j + 1):
        p = model.predict(p, pos[i], M)
    return p
