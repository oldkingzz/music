"""HMM + a SOFT guess of her next note from a melody n-gram trained on Bach sopranos AND ~8000
Essen folk melodies (music21 corpus, ABC), plus Repeat ('remember the song') when it fires.

Melody model: interpolated n-gram over her beat notes (relative pitch class per beat = longest note
in the beat), orders 0..N, keyed by bar position; Bach and Essen counts are added with weights.
The guess q(y) for beat k is fed to the HMM as a mixture likelihood  L(z) = (sum_y q(y) B[z, y])^w
(instead of a one-hot 'as if heard' note), via a ChordHMM subclass whose loglik understands a Soft
observation. Uniform q gives no evidence, a confident q acts like a heard note.

Causal: the context is soprano_obs(sop, j) for j < jstar fully and soprano_obs(sop, jstar, frac)
for the beat being heard at decision time k - D; nothing of beat k or later, no labels.
Essen sequences are parsed by a small ABC reader and cached in bench/cache/essen_seq.json.
"""
import json
import os
import re
import time
from collections import defaultdict

import numpy as np

from exp_ceiling import Repeat
from model import ChordHMM, soprano_obs
from planner import run

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, 'bench', 'cache', 'essen_seq.json')
CORPUS = os.path.join(HERE, '..', '.venv', 'lib', 'python3.11', 'site-packages', 'music21', 'corpus',
                      'essenFolksong')

# ----------------------------------------------------------------------------- variant knobs
ORDER = 3          # longest context (beats)
W_ESSEN = 1.0      # weight of Essen counts relative to Bach counts
BETA = 2.0         # interpolation strength towards the lower order
W_LL = 1.5         # exponent on the mixture likelihood
RATIO = False      # divide q by the marginal note distribution (product-of-experts form)
USE_REPEAT = True  # Repeat (last 2 beats match) overrides with its sharper guess
REP_CONF = 0.95    # Repeat's guess mass when it fires (rest = n-gram q)
SKIP_FILES = ('han1', 'han2')   # Essen files to leave out (e.g. Chinese 'han1', 'han2')
globals().update(json.loads(os.environ.get('ESSEN_CFG', '{}')))   # variant sweeps only

# ----------------------------------------------------------------------------- ABC reader
STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARPS, FLATS = 'FCGDAEB', 'BEADGCF'
NSIG = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'H': 5, 'F#': 6, 'F': -1, 'Bb': -2,
        'Eb': -3, 'Es': -3, 'Ab': -4, 'Db': -5}
TONIC = {'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'H': 11, 'F#': 6, 'F': 5, 'Bb': 10,
         'Eb': 3, 'Es': 3, 'Ab': 8, 'Db': 1}
TOK = re.compile(r"\(\d|\"[^\"]*\"|![^!]*!|\{[^}]*\}|(\^\^|\^|__|_|=)?([A-Ga-gz])([',]*)(\d*)(/*)(\d*)"
                 r"|[|:\]\[]+\d*|-|>|<|.", re.S)


def keysig(k):
    n = NSIG[k]
    return {c: 1 for c in SHARPS[:n]} if n >= 0 else {c: -1 for c in FLATS[:-n]}


def parse_tune(head, body):
    if head.get('M', 'none') in ('none', '') or head.get('K') not in NSIG:
        return None
    num, den = (int(x) for x in head['M'].split('/'))
    L = eval(head.get('L', '1/8').replace(' ', ''))
    if den >= 8 and num % 3 == 0 and num >= 6 or (num == 3 and den == 8):
        beat, M = 3.0 / den, max(1, num // 3)               # compound: dotted beats
    else:
        beat, M = 1.0 / den, num
    sig, tonic = keysig(head['K']), TONIC[head['K']]
    notes, t, acc, tie, trip, broken = [], 0.0, {}, False, [0, 1.0], None
    first_bar = None
    for m in TOK.finditer(body):
        s = m.group(0)
        if m.group(2):
            acc_s, letter, octs, d1, sl, d2 = m.groups()
            dur = L * (int(d1) if d1 else 1) / ((int(d2) if d2 else 2 ** len(sl)) if sl else 1)
            if trip[0] > 0:
                dur *= trip[1]
                trip[0] -= 1
            if broken is not None:
                dur *= broken
                broken = None
            if letter == 'z':
                notes.append([t, t + dur, None])
                t += dur
                tie = False
                continue
            up = letter.upper()
            key = (up, letter.islower(), octs)
            if acc_s:
                acc[key] = {'^': 1, '^^': 2, '_': -1, '__': -2, '=': 0}[acc_s]
            alt = acc.get(key, sig.get(up, 0))
            midi = 60 + STEP[up] + alt + (12 if letter.islower() else 0) + 12 * octs.count("'") \
                - 12 * octs.count(',')
            if tie and notes and notes[-1][2] == midi:
                notes[-1][1] = t + dur
            else:
                notes.append([t, t + dur, midi])
            t += dur
            tie = False
        elif s[0] in '|:][' and '|' in s:
            acc = {}
            if first_bar is None and t > 0:
                first_bar = t
        elif s == '-':
            tie = True
        elif s == '>' and notes:
            notes[-1][1] = notes[-1][0] + 1.5 * (notes[-1][1] - notes[-1][0])
            t = notes[-1][1]
            broken = 0.5
        elif s == '<' and notes:
            notes[-1][1] = notes[-1][0] + 0.5 * (notes[-1][1] - notes[-1][0])
            t = notes[-1][1]
            broken = 1.5
        elif s.startswith('(') and s[1:].isdigit():
            n = int(s[1:])
            trip = [n, {2: 1.5, 3: 2 / 3, 4: 0.75}.get(n, 2.0 / n)]
    if not notes or t <= 0:
        return None
    bar = num * (1.0 / den)
    shift = (bar - first_bar) if first_bar is not None and first_bar < bar - 1e-9 else 0.0
    sop = [[(on + shift) / beat, (off + shift) / beat, mi] for on, off, mi in notes if mi is not None]
    if not sop:
        return None
    b0 = int(np.floor(sop[0][0] + 1e-9))
    sop = [[on - b0, off - b0, mi] for on, off, mi in sop]
    K = int(np.ceil(sop[-1][1] - 1e-9))
    # mode: keep major (major third outweighs minor third over the tonic)
    dur = np.zeros(12)
    for on, off, mi in sop:
        dur[(mi - tonic) % 12] += off - on
    if dur[4] <= dur[3]:
        return None
    seq = []
    for k in range(K):
        mid = soprano_obs(sop, k)[1]
        seq.append(-1 if mid is None else (mid - tonic) % 12)
    return {'M': M, 'pos0': b0 % M, 'seq': seq}


def parse_corpus(budget=240.0):
    t0, out, n_all = time.time(), [], 0
    for fn in sorted(os.listdir(CORPUS)):
        if not fn.endswith('.abc') or fn.startswith('test'):
            continue
        txt = open(os.path.join(CORPUS, fn), encoding='latin-1').read()
        for chunk in re.split(r'\n(?=X:)', txt):
            if time.time() - t0 > budget:
                break
            head, body = {}, []
            for line in chunk.splitlines():
                mm = re.match(r'^([A-Z]):\s*(.*)$', line)
                if mm and not body:
                    head[mm.group(1)] = mm.group(2).strip()
                elif line.startswith('%') or mm:
                    continue
                else:
                    body.append(line)
            if 'X' not in head:
                continue
            n_all += 1
            try:
                r = parse_tune(head, '\n'.join(body))
            except Exception:
                r = None
            if r and len(r['seq']) >= 4:
                r['file'] = fn[:-4]
                out.append(r)
    return {'tunes': out, 'n_seen': n_all, 'secs': time.time() - t0}


def load_essen():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        d = parse_corpus()
        with open(CACHE, 'w') as fh:
            json.dump(d, fh, separators=(',', ':'))
    return json.load(open(CACHE))


# ----------------------------------------------------------------------------- melody model
def bach_seq(s):
    seq = []
    for k in range(s['K']):
        mid = soprano_obs(s['sop'], k)[1]
        seq.append(-1 if mid is None else (mid - s['tonic']) % 12)
    return seq, [b['pos'] for b in s['beats']]


class SoftNGram:
    def __init__(self, n, beta):
        self.n, self.beta = n, beta
        self.c = defaultdict(lambda: np.zeros(12))

    def add(self, seq, pos, w):
        for k in range(1, len(seq)):
            y = seq[k]
            if y < 0:
                continue
            for m in range(self.n + 1):
                ctx = tuple(seq[max(0, k - m):k])
                if len(ctx) < m:
                    break
                self.c[(m, pos[k], ctx)][y] += w
        return self

    def freeze(self):
        self.c = dict(self.c)
        tot = sum(v for (m, _, _), v in self.c.items() if m == 0)
        self.marg = (tot + 1.0) / (tot.sum() + 12.0)
        return self

    def dist(self, ctx_full, pos):
        p = self.marg.copy()
        for m in range(self.n + 1):
            ctx = tuple(ctx_full[len(ctx_full) - m:]) if m else ()
            if len(ctx) < m:
                break
            c = self.c.get((m, pos, ctx))
            if c is None:
                break
            p = (c + self.beta * p) / (c.sum() + self.beta)
        return p


class Soft:
    """A guessed observation; survives planner.to_obs's (mid - tonic) % 12."""
    def __init__(self, q, w):
        self.q, self.w = q, w

    def __sub__(self, o):
        return self

    def __mod__(self, o):
        return self


class SoftHMM(ChordHMM):
    def loglik(self, obs, mode, lam=1.0):
        if isinstance(obs, Soft):
            return obs.w * np.log(self.B @ obs.q + 1e-12)
        return super().loglik(obs, mode, lam)


class System:
    def __init__(self, train):
        base = ChordHMM().fit(train['bach'])
        self.model = SoftHMM.__new__(SoftHMM)
        self.model.__dict__.update(base.__dict__)
        self.ng = SoftNGram(ORDER, BETA)
        for s in train['bach']:
            self.ng.add(*bach_seq(s), 1.0)
        ess = load_essen()
        self.n_essen = 0
        if W_ESSEN > 0:
            for r in ess['tunes']:
                if r['file'] in SKIP_FILES:
                    continue
                pos = [(r['pos0'] + i) % r['M'] for i in range(len(r['seq']))]
                self.ng.add(r['seq'], pos, W_ESSEN)
                self.n_essen += 1
        self.ng.freeze()
        self.note_guess = {}

    def run(self, song, cfg):
        t = song['tonic']
        pos = [b['pos'] for b in song['beats']]
        D = (cfg.L + cfg.delta) / (60.0 / cfg.bpm)
        rep = Repeat(song, 2, D) if USE_REPEAT else None
        heard = {}

        def obs(j, frac):
            key = (j, round(frac, 6))
            if key not in heard:
                mid = soprano_obs(song['sop'], j, frac)[1]
                heard[key] = -1 if mid is None else (mid - t) % 12
            return heard[key]

        def hint(k, jstar, frac):
            if jstar < 1:
                return None
            ctx = [obs(j, 1.0) for j in range(max(0, jstar - ORDER), jstar)] + [obs(jstar, frac)]
            ctx += [-1] * (k - jstar - 1)                 # unheard beats between (none at this D)
            q = self.ng.dist(ctx, pos[k])
            if rep is not None:
                g = rep(k, jstar, frac)
                if g is not None and g[1] is not None:
                    q = (1 - REP_CONF) * q
                    q[(g[1] - t) % 12] += REP_CONF
            self.note_guess[k] = int(np.argmax(q))
            if RATIO:
                q = q / self.ng.marg
                q = q / q.sum()
            return np.zeros(12), Soft(q, W_LL)
        return run(song, self.model, cfg, 'hmm', hint=hint)


def build(train):
    return System(train)


if __name__ == '__main__':
    d = load_essen()
    print(len(d['tunes']), 'major tunes kept of', d['n_seen'], 'seen in', round(d['secs'], 1), 's')
