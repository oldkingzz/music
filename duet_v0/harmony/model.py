"""Learned harmony line for Duet: a linear-chain CRF trained on Bach's chorales.

Drop-in replacement for follow.plan_harmony:

    from harmony.model import plan_harmony_learned
    plan, key_name = plan_harmony_learned(notes)      # notes from follow.offline_notes()

`plan` is a list of dicts {'note', 'r', 'dur', 'midi', 'h'} exactly like follow.plan_harmony
(same notes kept: dur >= min_dur and midi != 0), so Accompanist(notes, plan, ...) works as is.

How it works (all numpy, CPU, ~ms per song):
  1. Key from the melody alone: Krumhansl-Schmuckler correlation with Aarden-Essen profiles
     plus a small bonus when the last note is the tonic (key agreement with music21's
     full-score key on the held-out test chorales: 83 % vs 64 % for follow.detect_key).
  2. Transpose to a C frame (tonic -> C, nearest direction) and shift by octaves so the
     melody sits in Bach's soprano register; all statistics live in that frame.
  3. Hidden state per melody note = interval of the harmony voice below the melody,
     0..16 semitones (alto). The CRF scores a whole line with learned weights on
       unary:    interval | scale degree, mode, phrase end / start, note length, next degree;
                 absolute register of the harmony note
       pairwise: harmony motion | melody motion; interval pair | melody direction
                 (this is where parallel fifths/octaves are learned); harmony degree pair;
                 harmony motion | previous harmony degree (leading-tone resolution).
  4. Viterbi decoding, optionally with duet-specific constraints: forbid parallel perfect
     fifths / octaves with the melody, and a penalty on intervals that are dissonant in a
     two-voice texture (2nds, 4th, tritone, 7ths) - Bach writes 4ths between S and A because
     tenor and bass complete the chord; Duet has only two voices - and a penalty on holding
     the same harmony note (Viterbi otherwise over-produces held notes).
  5. Transpose back.

Training: harmony/train.py. Evaluation: harmony/evaluate.py, results in harmony/eval.md.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, 'params', 'crf_{voice}.npz')
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Aarden-Essen key profiles (as shipped in music21.analysis.discrete.AardenEssen)
AE_MAJOR = np.array([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248,
                     22.062, 0.145624, 8.15494, 0.232998, 4.95122])
AE_MINOR = np.array([18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494,
                     18.6161, 4.56621, 1.93186, 7.37619, 1.75623])
FINAL_TONIC_BONUS = 0.2          # chosen on the training split (see eval.md, key detection)

# interval range (semitones below the melody) per voice; positions whose true interval falls
# outside are treated as unlabeled during training
VOICE_RANGE = {'alto': (0, 16), 'tenor': (3, 28), 'bass': (7, 40)}
VOICE_COL = {'alto': 4, 'tenor': 5, 'bass': 6}            # column in chorales.py rows
DISSONANT_2V = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], bool)   # m2 M2 P4 TT m7 M7
GAP_PHRASE = 0.10                # s; a breath of at least this long ends a phrase


# ---------------------------------------------------------------- key and context
def detect_key_melody(notes, bonus=FINAL_TONIC_BONUS):
    """(tonic pc, 'major'|'minor') from a melody [{'on','off','midi'}]."""
    h = np.zeros(12)
    voiced = [nt for nt in notes if nt['midi'] > 0]
    for nt in voiced:
        h[nt['midi'] % 12] += max(nt['off'] - nt['on'], 0.0)
    if not voiced or h.std() == 0:
        return 0, 'major'
    last = voiced[-1]['midi'] % 12
    best = None
    for t in range(12):
        for mode, prof in (('major', AE_MAJOR), ('minor', AE_MINOR)):
            c = np.corrcoef(h, np.roll(prof, t))[0, 1] + bonus * (last == t)
            if best is None or c > best[0]:
                best = (c, t, mode)
    return best[1], best[2]


def frame_shift(tonic):
    """Semitones that move `tonic` to C by the shorter way (-6..+5)."""
    return ((-tonic + 6) % 12) - 6


def context(notes, tonic, mode, sop_median, min_dur=0.12):
    """Per kept note: frame pitch s, degree d, flags. Returns (kept indices, ctx dict, shift)."""
    keep = [k for k, nt in enumerate(notes) if nt['off'] - nt['on'] >= min_dur and nt['midi'] != 0]
    if not keep:
        return keep, None, 0
    T = len(keep)
    raw = np.array([notes[k]['midi'] for k in keep], float)
    sh = frame_shift(tonic)
    octv = int(np.round((sop_median - (np.median(raw) + sh)) / 12.0))
    shift = sh + 12 * octv
    s = (raw + shift).astype(int)
    dur = np.array([notes[k]['off'] - notes[k]['on'] for k in keep])
    med = np.median(dur)
    durc = np.where(dur < 0.75 * med, 0, np.where(dur > 1.6 * med, 2, 1))
    end = np.zeros(T, int)
    beg = np.zeros(T, int)
    beg[0] = 1
    end[-1] = 1
    for j, k in enumerate(keep):                     # a gap after the note (before the next
        nxt = next((n for n in notes[k + 1:] if n['midi'] != 0), None)   # voiced note) = breath
        if nxt is not None and nxt['on'] - notes[k]['off'] >= GAP_PHRASE:
            end[j] = 1
            if j + 1 < T:
                beg[j + 1] = 1
    d = s % 12
    dnext = np.append(d[1:], 12)
    ds = np.append(0, np.diff(s))
    m = 0 if mode == 'major' else 1
    return keep, {'s': s, 'd': d, 'dur': durc, 'end': end, 'beg': beg, 'dnext': dnext,
                  'ds': ds, 'mode': m}, shift


# ---------------------------------------------------------------- features
class Templates:
    """Feature templates. Every (position, state) [unary] or (position, prev, state) [pair]
    maps to one weight per template; w is the concatenation of all template tables.
    `features` selects a subset (used for the ablation in eval.md)."""

    UNARY = ('iv|mode,deg', 'ivpc|mode,end,deg', 'ivpc|mode,len', 'register',
             'ivpc|mode,deg,nextdeg', 'ivpc|mode,start,deg', 'ivpc|mode,prevdeg,deg', 'iv|mode,len')
    PAIR = ('dh|ds', 'ivpc,ivpc|dir', 'hdeg,hdeg|mode', 'iv,iv', 'dh|mode,hdeg', 'ivpc,ivpc|ds')
    DEFAULT = UNARY[:6] + PAIR[:5]

    def __init__(self, lo, hi, features=None):
        self.lo, self.hi = lo, hi
        self.K = K = hi - lo + 1
        size = {'iv|mode,deg': 2 * 12 * K, 'ivpc|mode,end,deg': 2 * 2 * 12 * 12,
                'ivpc|mode,len': 2 * 3 * 12, 'register': 61,
                'ivpc|mode,deg,nextdeg': 2 * 12 * 13 * 12, 'ivpc|mode,start,deg': 2 * 2 * 12 * 12,
                'ivpc|mode,prevdeg,deg': 2 * 13 * 12 * 12, 'iv|mode,len': 2 * 3 * K,
                'dh|ds': 25 * 15, 'ivpc,ivpc|dir': 3 * 12 * 12, 'hdeg,hdeg|mode': 2 * 12 * 12,
                'iv,iv': K * K, 'dh|mode,hdeg': 2 * 12 * 25, 'ivpc,ivpc|ds': 15 * 12 * 12}
        feats = list(features or self.DEFAULT)
        self.unary = [f for f in self.UNARY if f in feats]
        self.pair = [f for f in self.PAIR if f in feats]
        self.names = self.unary + self.pair
        self.off = np.concatenate([[0], np.cumsum([size[n] for n in self.names])]).astype(int)
        self.W = int(self.off[-1])

    def slices(self):
        return {name: slice(self.off[i], self.off[i + 1]) for i, name in enumerate(self.names)}

    def index(self, c):
        """uidx (nU, T, K) and pidx (nP, T-1, K, K) int32 global weight indices."""
        K, lo = self.K, self.lo
        s, d, m = c['s'], c['d'], c['mode']
        T = len(s)
        iv = np.arange(lo, self.hi + 1)[None, :]                       # (1, K)
        k = iv - lo
        ivpc = iv % 12
        h = s[:, None] - iv                                             # harmony pitch (T, K)
        col = lambda x: np.asarray(x)[:, None]                          # noqa: E731
        dprev = np.append(12, d[:-1])
        loc = {'iv|mode,deg': lambda: (m * 12 + col(d)) * K + k,
               'ivpc|mode,end,deg': lambda: ((m * 2 + col(c['end'])) * 12 + col(d)) * 12 + ivpc,
               'ivpc|mode,len': lambda: (m * 3 + col(c['dur'])) * 12 + ivpc,
               'register': lambda: np.clip(h, 30, 90) - 30,
               'ivpc|mode,deg,nextdeg': lambda: ((m * 12 + col(d)) * 13 + col(c['dnext'])) * 12 + ivpc,
               'ivpc|mode,start,deg': lambda: ((m * 2 + col(c['beg'])) * 12 + col(d)) * 12 + ivpc,
               'ivpc|mode,prevdeg,deg': lambda: ((m * 13 + col(dprev)) * 12 + col(d)) * 12 + ivpc,
               'iv|mode,len': lambda: (m * 3 + col(c['dur'])) * K + k}
        U = np.empty((len(self.unary), T, K), np.int32)
        for i, n in enumerate(self.unary):
            U[i] = self.off[i] + np.broadcast_to(loc[n](), (T, K))
        nU = len(self.unary)
        if T < 2:
            return U, np.zeros((len(self.pair), 0, K, K), np.int32)
        hp, hc = h[:-1, :, None], h[1:, None, :]                        # (T-1, K, 1), (T-1, 1, K)
        dh = np.clip(hc - hp, -12, 12) + 12
        ds = np.clip(c['ds'][1:], -7, 7)[:, None, None]
        sgn = np.sign(ds) + 1
        ip, ic = ivpc[0][None, :, None], ivpc[0][None, None, :]
        kp, kc = k[0][None, :, None], k[0][None, None, :]
        ploc = {'dh|ds': lambda: (ds + 7) * 25 + dh,
                'ivpc,ivpc|dir': lambda: (sgn * 12 + ip) * 12 + ic,
                'hdeg,hdeg|mode': lambda: (m * 12 + hp % 12) * 12 + hc % 12,
                'iv,iv': lambda: kp * K + kc,
                'dh|mode,hdeg': lambda: (m * 12 + hp % 12) * 25 + dh,
                'ivpc,ivpc|ds': lambda: ((ds + 7) * 12 + ip) * 12 + ic}
        P = np.empty((len(self.pair), T - 1, K, K), np.int32)
        for i, n in enumerate(self.pair):
            P[i] = self.off[nU + i] + np.broadcast_to(ploc[n](), (T - 1, K, K))
        return U, P


# ---------------------------------------------------------------- inference
def potentials(w, U, P):
    return w[U].sum(0), (w[P].sum(0) if P.shape[1] else np.zeros((0,) + U.shape[2:] * 2))


def duet_masks(c, tpl, lam=0.0, no_parallel=True, max_iv=None, beta=0.0):
    """Extra log-potentials for the two-voice duet (added to the learned ones at decoding).

    lam   penalty on two-voice dissonances (x1.5 on long notes, +1 at phrase ends: a 4th or
          2nd that a full choir would cover sounds exposed in a duet, worst at cadences)
    beta  penalty on repeating the previous harmony pitch. MAP (Viterbi) decoding of a
          sequence model over-produces its most probable move, which here is "hold the note"
          (63 % repeated notes vs Bach's 38 % on validation); beta re-calibrates that.
    """
    T, K = len(c['s']), tpl.K
    iv = np.arange(tpl.lo, tpl.hi + 1)
    u = np.zeros((T, K))
    weight = 1.0 + 0.5 * (c['dur'] == 2) + 1.0 * c['end']           # long notes, phrase ends
    u -= lam * DISSONANT_2V[iv % 12][None, :] * weight[:, None]
    if max_iv is not None:
        u[:, iv > max_iv] = -np.inf
    p = np.zeros((max(T - 1, 0), K, K))
    if no_parallel and T > 1:
        same = (iv[:, None] % 12 == iv[None, :] % 12) & np.isin(iv % 12, (0, 7))[None, :]
        moving = c['ds'][1:] != 0
        p[moving] += np.where(same, -np.inf, 0.0)[None]
    if beta and T > 1:
        hold = (iv[None, None, :] - iv[None, :, None]) == c['ds'][1:, None, None]
        p -= beta * hold
    return u, p


def viterbi(u, p):
    T, K = u.shape
    delta = u[0].copy()
    back = np.zeros((T, K), int)
    for t in range(1, T):
        sc = delta[:, None] + p[t - 1]
        back[t] = np.argmax(sc, 0)
        delta = sc[back[t], np.arange(K)] + u[t]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class HarmonyModel:
    def __init__(self, w, cfg):
        self.w = np.asarray(w, float)
        self.cfg = cfg
        self.tpl = Templates(*cfg['iv_range'], cfg.get('features'))

    @classmethod
    def load(cls, voice='alto', path=None):
        z = np.load(path or PARAMS.format(voice=voice), allow_pickle=False)
        return cls(z['w'], json.loads(str(z['cfg'])))

    def save(self, path):
        np.savez(path, w=self.w, cfg=json.dumps(self.cfg))

    def decode(self, c, lam=None, no_parallel=None, max_iv=None, beta=None):
        """Interval below the melody for every kept note (array, semitones)."""
        dflt = self.cfg.get('duet', {})
        beta = dflt.get('beta', 0.0) if beta is None else beta
        lam = dflt.get('lam', 0.0) if lam is None else lam
        no_parallel = dflt.get('no_parallel', True) if no_parallel is None else no_parallel
        max_iv = dflt.get('max_iv') if max_iv is None else max_iv
        U, P = self.tpl.index(c)
        u, p = potentials(self.w, U, P)
        du, dp = duet_masks(c, self.tpl, lam, no_parallel, max_iv, beta)
        u, p = u + du, p + dp
        if not np.isfinite(u.max(1)).all():
            u = np.where(np.isfinite(u).any(1, keepdims=True), u, 0.0)
        return viterbi(u, p) + self.tpl.lo

    def harmonize(self, notes, min_dur=0.12, key=None, **kw):
        tonic, mode = key if key is not None else detect_key_melody(notes)
        keep, c, shift = context(notes, tonic, mode, self.cfg['sop_median'], min_dur)
        if not keep:
            return [], f'{NAMES[tonic]} {mode}'
        ivs = self.decode(c, **kw)
        plan = []
        for j, k in enumerate(keep):
            nt = notes[k]
            plan.append({'note': k, 'r': nt['on'], 'dur': nt['off'] - nt['on'], 'midi': nt['midi'],
                         'h': int(nt['midi'] - ivs[j])})
        return plan, f'{NAMES[tonic]} {mode}'


_CACHE = {}


def plan_harmony_learned(notes, min_dur=0.12, voice='alto', **kw):
    """Same contract as follow.plan_harmony: returns (plan, key_name)."""
    if voice not in _CACHE:
        _CACHE[voice] = HarmonyModel.load(voice)
    return _CACHE[voice].harmonize(notes, min_dur=min_dur, **kw)
