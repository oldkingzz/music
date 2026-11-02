"""causal_probe (verifier, run v5rt_1101): is rt:r2_copyhmbold_v1 causal under bench_rt? Val only (tuned on val; this
bundle has no test split). Uses bench_rt.load / simulate / HEARD1 unchanged. Writes ONE new JSON (mode 'x').

Songs: 24 val songs, 6 per dataset, random.Random(SEED).sample per dataset (fixed). Heard melody = bench3's B2.heard
(seed 1), exactly as bench_rt.evaluate builds it. A recorder around the system logs every decision: (time, kind, the
inputs bench_rt passed, the return value).

T1 cut (the task's spec), L = 0.1: 3 cut times per song (near 30 % / 55 % / 80 % of her notes: snapped to a note onset,
   to a plan point, and off the grid). Copies keep every true note with onset < c; a note sounding across c gets a
   random new offset > c; every later note is replaced by a random melody (pitches, onsets, durations); the chords of
   beats >= int(c) are replaced at random. 2 random copies per cut; whole songs are simulated. PASS = every decision
   at a time < c identical (also reported for <= c) and the inputs bench_rt gave before c identical.
T2 latency cut, L = 0.05 / 0.1 / 0.15: as T1, but everything not yet HEARD at c is replaced (notes with onset + L > c,
   i.e. starting after c - L, and an offset not heard by c). PASS = every decision at a time <= c identical.
T3 plan(q) vs unheard pitches: up to 30 plan points per song at L = 0.1 (8 at 0.05 / 0.15), mostly where a note starts
   in (q - L, q + 0.01); the true pitch of every note with onset + L > q (= starting at or after q - L; notes with
   onset + L == q exactly are counted: they are heard at q) is changed, timing kept. The run stops after q (decisions
   are made in time order, so later inputs cannot reach back; run history is T4's job). PASS = plan(q) and every
   earlier decision identical.
T4 run history: the 24 original songs' full decision logs from (a) object A's first pass, (b) object A again after all
   T1-T3 runs, (c) a freshly built object B fed the songs in reverse order. PASS = identical (no memory across runs or
   songs; without this T1-T3 could not see a system that memorises a song between runs).
T5 tripwires: during every simulate() call, calls to open / io.open / os.open / np.load / torch.load / json.load(s), the
   bench data loaders (bench3 + bench train_sets / eval_sets / heard / _extra / data.load, bench_rt HEARD1 /
   heard_seed2) and os.environ reads are logged with the calling project file. Build-time reads are listed apart.
T6 power: a toy that memorises songs across runs must fail T4/T1; a toy that reads os.environ / bench data in plan()
   must trip T5.

  .venv/bin/python runs_mini/v5rt_1101/verify/causal_probe.py --out runs_mini/v5rt_1101/verify/<new>.json
"""
import argparse
import ast
import builtins
import copy
import io
import json
import os
import random
import sys
import time

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v5_rt'))
import bench_rt as B  # noqa: E402  (puts the v3 / duet_beat / rt_systems paths in place)
import numpy as np  # noqa: E402
import torch  # noqa: E402

SPEC = 'rt:r2_copyhmbold_v1'
SEED = 20261101
DS = ('bach', 'folk', 'pop', 'pop909')
HERE = os.path.abspath(__file__)
CHAIN = ['futures/v5_rt/rt_systems/r2_copyhmbold_v1.py', 'futures/v5_rt/rt_systems/r2_lib_v3.py',
         'futures/v5_rt/rt_systems/r2_lib_v2.py', 'futures/v5_rt/rt_systems/r2_lib_v1.py',
         'futures/v5_rt/rt_systems/r1_A_core_v3.py', 'futures/v5_rt/rt_systems/r1_A_core_v2.py',
         'futures/v3_nolatency/systems/r2_mix_v2.py', 'futures/v3_nolatency/systems/m1_filter_v5lo.py',
         'futures/v3_nolatency/systems/m3_seq_v4.py', 'futures/v3_nolatency/systems/m3_seq_v1.py']


# ------------------------------------------------------------------------------------------------ tripwires (T5)
class Trip:
    def __init__(self):
        self.phase = None
        self.hits = {}                       # (phase, name, innermost project frame, immediate caller file) -> n
        self.paths = {}                      # build-time file arguments
        self.busy = False
        self.saved = []

    def hit(self, name, args):
        if self.phase is None or self.busy:
            return
        self.busy = True
        try:
            f = sys._getframe(2)
            caller = f.f_code.co_filename if f else '?'
            where = '-'
            while f is not None:
                fn = f.f_code.co_filename
                if fn.startswith(ABS):
                    where = f"{os.path.relpath(fn, ABS)}:{f.f_lineno}"
                    break
                f = f.f_back
            key = (self.phase, name, where, os.path.relpath(caller, ABS) if caller.startswith(ABS) else caller)
            self.hits[key] = self.hits.get(key, 0) + 1
            if self.phase == 'build' and args and isinstance(args[0], (str, bytes, os.PathLike)):
                p = os.fspath(args[0])
                p = p.decode() if isinstance(p, bytes) else p
                self.paths[(name, p)] = self.paths.get((name, p), 0) + 1
        finally:
            self.busy = False

    def _wrap(self, name, fn):
        trip = self

        def w(*a, **k):
            trip.hit(name, a)
            return fn(*a, **k)
        return w

    def install(self):
        b3, b2 = B.bench3, B.bench3.B2
        data = sys.modules.get('data')
        targets = [(builtins, 'open'), (io, 'open'), (os, 'open'), (np, 'load'), (torch, 'load'), (json, 'load'),
                   (json, 'loads'), (b3, 'train_sets'), (b3, 'eval_sets'), (b3, 'run_song'), (b2, 'train_sets'),
                   (b2, 'eval_sets'), (b2, 'heard'), (b2, '_extra'), (b2, 'load'), (B, 'HEARD1'), (B, 'heard_seed2'),
                   (B, 'evaluate'), (B, 'score_song')]
        if data is not None and hasattr(data, 'load'):
            targets.append((data, 'load'))
        for mod, attr in targets:
            fn = getattr(mod, attr)
            self.saved.append((mod, attr, fn))
            setattr(mod, attr, self._wrap(f'{getattr(mod, "__name__", mod)}.{attr}', fn))
        gi = os._Environ.__getitem__
        self.saved.append((os._Environ, '__getitem__', gi))
        trip = self

        def env_get(env, key):
            trip.hit(f'os.environ[{key!r}]', ())
            return gi(env, key)
        os._Environ.__getitem__ = env_get

    def uninstall(self):
        for mod, attr, fn in reversed(self.saved):
            setattr(mod, attr, fn)
        self.saved = []


TRIP = Trip()


# ------------------------------------------------------------------------------------------------ recorder
class Stop(Exception):
    pass


class Rec:
    """Logs every decision: (time, 0 plan / 1 hear, inputs, return value or None). stop: raise once time > stop."""

    def __init__(self, obj, L):
        self.obj, self.L, self.stop, self.log = obj, L, None, []

    def start(self, meta):
        self.log = []
        self.obj.start(meta)

    def plan(self, q, heard):
        if self.stop is not None and q > self.stop:
            raise Stop
        h = self.obj.plan(q, heard)
        self.log.append((q, 0, tuple(heard), None if h is None else int(h)))
        return h

    def hear(self, i, onset, midi, prev_dur):
        t = onset + self.L                       # bench_rt's hear time: the same float
        if self.stop is not None and t > self.stop:
            raise Stop
        h = self.obj.hear(i, onset, midi, prev_dur)
        self.log.append((t, 1, (i, onset, midi, prev_dur), None if h is None else int(h)))
        return h


def run(rec, meta, notes, L, stop=None):
    """bench_rt.simulate (unchanged) under the tripwires; returns (log, error or None)."""
    rec.stop = stop
    err = None
    TRIP.phase = 'run'
    try:
        B.simulate(rec, meta, notes, L)
    except Stop:
        pass
    except Exception as e:  # noqa: BLE001  (recorded: a crash on a random future is a robustness note)
        err = f'{type(e).__name__}: {e}'
    finally:
        TRIP.phase = None
        rec.stop = None
    return list(rec.log), err


def heard_notes(song):
    return [(float(on), float(off), int(m)) for on, off, m in sorted(B.HEARD1(song)['sop'], key=lambda n: n[0])]


def meta_of(song):
    return {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'pos': [b['pos'] for b in song['beats']]}


def upto(log, c, strict):
    return [e for e in log if (e[0] < c if strict else e[0] <= c)]


def compare(la, lb, c, strict):
    a, b = upto(la, c, strict), upto(lb, c, strict)
    inp = [(e[0], e[1], e[2]) for e in a] == [(e[0], e[1], e[2]) for e in b]
    dec = [(e[0], e[1], e[3]) for e in a] == [(e[0], e[1], e[3]) for e in b]
    first = None
    if not dec:
        for x, y in zip(a, b):
            if (x[0], x[1], x[3]) != (y[0], y[1], y[3]):
                first = {'t': x[0], 'kind': 'plan' if x[1] == 0 else 'hear', 'orig': x[3], 'copy': y[3]}
                break
        if first is None:
            first = {'len_orig': len(a), 'len_copy': len(b)}
    return inp, dec, len(a), sum(e[3] is not None for e in a), first


# ------------------------------------------------------------------------------------------------ perturbations
def rand_future(rng, t0, K, lo, hi):
    out, t = [], t0
    while t < K - 0.05:
        d = rng.choice((0.25, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0, round(rng.uniform(0.05, 2.5), 3)))
        out.append([t, t + d, rng.randint(lo, hi)])
        t += d + (0.0 if rng.random() < 0.6 else rng.choice((0.25, 0.5, 1.0, round(rng.uniform(0.01, 1.5), 3))))
    return out


def pitch_range(song):
    ms = [m for _, _, m in song['sop']]
    return max(40, min(ms) - 7), min(95, max(ms) + 7)


def copy_cut(song, c, L, rng, latency):
    """latency False (T1): keep true notes with onset < c, replace the rest; True (T2): keep notes heard by c
    (onset + L <= c), replace the rest. Returns the song copy and the kept-note count."""
    s2 = copy.deepcopy(song)
    keep = (lambda n: float(n[0]) + L <= c) if latency else (lambda n: float(n[0]) < c)
    kept = [list(n) for n in song['sop'] if keep(n)]
    unseen_off = (lambda off: float(off) + L > c) if latency else (lambda off: float(off) > c)
    floor_on = c - L if latency else c                  # new onsets must be > c - L (T2) or >= c (T1)
    t_free = floor_on
    if kept:
        p = kept[-1]
        if unseen_off(p[1]):                            # its end is not known by c: new random end, still unseen
            lo_end = max(float(p[0]) + 0.01, (c - L + 0.001) if latency else (c + 0.001))
            p[1] = lo_end + rng.choice((0.0, 0.25, 0.5, 1.0, round(rng.uniform(0.0, 2.0), 3)))
        t_free = max(t_free, float(p[1]))
    if latency:
        cand = rng.choice((c - L / 2, c, c + 0.25, c - L + 0.001, c + round(rng.uniform(0, 1.0), 3)))
        t0 = max(cand, t_free, c - L + 0.001)
    else:
        cand = rng.choice((c, c, c + 0.25, c + round(rng.uniform(0, 1.0), 3)))
        t0 = max(cand, t_free)
    lo, hi = pitch_range(song)
    s2['sop'] = kept + rand_future(rng, t0, song['K'], lo, hi)
    k0 = int(c - L) if latency else int(c)
    for k in range(max(k0, 0), len(s2['beats'])):
        s2['beats'][k]['z'] = rng.randint(-1, 6)
    return s2, len(kept)


def copy_pitch(song, q, L, rng):
    """T3: change the true pitch of every note with onset + L > q (unheard at q), timing kept."""
    s2 = copy.deepcopy(song)
    n_changed = n_tie = 0
    for n in s2['sop']:
        if float(n[0]) + L == q:
            n_tie += 1
        if float(n[0]) + L > q:
            n[2] = n[2] + rng.choice([d for d in range(-12, 13) if d != 0])
            n_changed += 1
    return s2, n_changed, n_tie


def prefix_ok(n0, n2, c, L, latency):
    """The copy's heard notes known by c must equal the original's (else the copy was built wrongly)."""
    known = (lambda n: n[0] + L <= c) if latency else (lambda n: n[0] < c)
    a = [n for n in n0 if known(n)]
    b = [n for n in n2 if known(n)]
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[0] != y[0] or x[2] != y[2]:
            return False
        off_known = (x[1] + L <= c) if latency else (x[1] <= c)
        if off_known and x[1] != y[1]:
            return False
    return True


def cut_times(n0, K, rng):
    ons = [n[0] for n in n0]
    first, last = ons[0], ons[-1]
    span = max(last - first, 1.0)
    ca = ons[min(len(ons) - 1, max(1, int(0.30 * (len(ons) - 1))))]          # a note onset
    cb = 0.5 * round((first + 0.55 * span) / 0.5)                              # a plan point
    cc = first + 0.80 * span + rng.uniform(0.013, 0.487)                       # off the grid
    while any(abs(cc - o) < 1e-6 for o in ons) or abs(cc / 0.5 - round(cc / 0.5)) < 1e-6:
        cc += 0.0071
    return [('onset', ca), ('grid', cb), ('offgrid', min(cc, K - 0.01))]


# ------------------------------------------------------------------------------------------------ toys (T6)
class ToyMemo:
    """Cheat: remembers every song it has heard (by tonic, M, K) and plans the remembered next pitch - 3."""
    mem = {}

    def start(self, meta):
        self.key, self.seen = (meta['tonic'], meta['M'], meta['K']), []

    def plan(self, q, heard):
        old = ToyMemo.mem.get(self.key)
        if not old:
            return None
        nxt = [m for on, m in old if on >= q]
        return nxt[0] - 3 if nxt else None

    def hear(self, i, onset, midi, prev_dur):
        self.seen.append((onset, midi))
        ToyMemo.mem[self.key] = list(self.seen) if len(self.seen) > len(ToyMemo.mem.get(self.key, [])) \
            else ToyMemo.mem[self.key]
        return midi - 4


class ToyPeek:
    """Cheat: reads os.environ and the bench's heard() inside plan()."""

    def start(self, meta):
        self.done = False

    def plan(self, q, heard):
        if not self.done:
            self.done = True
            os.environ.get('HOME')
            B.bench3.B2.heard({'id': 'x', 'sop': [], 'beats': []})
        return None

    def hear(self, i, onset, midi, prev_dur):
        return midi - 4


# ------------------------------------------------------------------------------------------------ static read
def static_scan():
    out = {}
    for rel in CHAIN:
        with open(os.path.join(ABS, rel)) as fh:
            tree = ast.parse(fh.read())
        imps, sus = set(), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imps.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imps.add(f"{node.module}:{','.join(a.name for a in node.names)}")
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in ('environ', 'getenv', 'open', 'load', 'loads', 'eval_sets', 'train_sets', 'heard', 'B2',
                        'bench', 'bench3', 'bench_rt', 'sop', 'beats'):
                sus.setdefault(name, []).append(node.lineno)
        out[rel] = {'imports': sorted(imps), 'names': sus}
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--nq', type=int, default=30)
    ap.add_argument('--nq_other', type=int, default=8)
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    if not out_path.startswith(os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify') + os.sep):
        sys.exit('--out must be under runs_mini/v5rt_1101/verify/')
    if os.path.exists(out_path):
        sys.exit(f'refusing to overwrite {out_path}')
    t_start = time.time()
    res = {'spec': SPEC, 'seed': SEED, 'split': 'val', 'bench_sha256': None, 'tests': {}, 'fail_examples': []}
    with open(B.__file__, 'rb') as fh:
        import hashlib
        res['bench_sha256'] = hashlib.sha256(fh.read()).hexdigest()
    res['static'] = static_scan()

    B.bench3.B2.heard = B.HEARD1                       # as bench_rt.evaluate does (seed 1)
    train = B.bench3.train_sets()
    VAL = B.bench3.eval_sets('val')
    assert list(VAL) == list(DS), list(VAL)
    TRIP.install()
    try:
        TRIP.phase = 'build'
        objA = B.load(SPEC, train)
        TRIP.phase = None
        rng = random.Random(SEED)
        pick = {ds: sorted(rng.sample(range(len(VAL[ds])), 6)) for ds in DS}
        songs = [(ds, j, VAL[ds][j]) for ds in DS for j in pick[ds]]
        res['songs'] = [{'ds': ds, 'idx': j, 'id': str(s.get('id', s.get('title', '')))[:40] if ds == 'bach' else j,
                         'notes': len(s['sop']), 'K': s['K']} for ds, j, s in songs]
        res['sop_sorted_all'] = all(all(s['sop'][i][0] <= s['sop'][i + 1][0] for i in range(len(s['sop']) - 1))
                                    for _, _, s in songs)
        recA = Rec(objA, 0.1)

        # originals (pass 1) at each L
        orig = {}
        for L in (0.05, 0.1, 0.15):
            recA.L = L
            for ds, j, s in songs:
                log, err = run(recA, meta_of(s), heard_notes(s), L)
                assert err is None, (ds, j, L, err)
                orig[(ds, j, L)] = log
        res['orig_decisions_L0.1'] = sum(len(orig[(ds, j, 0.1)]) for ds, j, _ in songs)
        res['orig_nonnull_L0.1'] = sum(sum(e[3] is not None for e in orig[(ds, j, 0.1)]) for ds, j, _ in songs)
        res['orig_plan_attacks_L0.1'] = sum(sum(e[1] == 0 and e[3] is not None for e in orig[(ds, j, 0.1)])
                                            for ds, j, _ in songs)

        def tally(name, ds, ok_inp, ok_dec, n, nn, extra=None):
            t = res['tests'].setdefault(name, {'cases': 0, 'inp_mismatch': 0, 'dec_mismatch': 0, 'decisions': 0,
                                                'nonnull_decisions': 0, 'by_ds': {}})
            t['cases'] += 1
            t['inp_mismatch'] += not ok_inp
            t['dec_mismatch'] += not ok_dec
            t['decisions'] += n
            t['nonnull_decisions'] += nn
            bd = t['by_ds'].setdefault(ds, [0, 0])
            bd[0] += 1
            bd[1] += not ok_dec
            if extra:
                for k, v in extra.items():
                    t[k] = t.get(k, 0) + v

        def fail(**kw):
            if len(res['fail_examples']) < 60:
                res['fail_examples'].append(kw)

        # T1 (spec cut, L = 0.1) and T2 (latency cut, 3 L)
        for test, Ls, n_fut in (('T1_cut', (0.1,), 2), ('T2_latency_cut', (0.05, 0.1, 0.15), 2)):
            for L in Ls:
                recA.L = L
                for ds, j, s in songs:
                    n0 = heard_notes(s)
                    r2 = random.Random(f'{SEED}-{test}-{L}-{ds}-{j}')
                    for ctype, c in cut_times(n0, s['K'], r2):
                        for f in range(n_fut):
                            s2, nk = copy_cut(s, c, L, r2, latency=(test != 'T1_cut'))
                            n2 = heard_notes(s2)
                            if not prefix_ok(n0, n2, c, L, latency=(test != 'T1_cut')):
                                tally(f'{test}_L{L}', ds, True, True, 0, 0, {'construction_mismatch': 1})
                                fail(test=test, L=L, ds=ds, idx=j, cut=c, why='construction mismatch')
                                continue
                            log2, err = run(recA, meta_of(s2), n2, L)
                            if test == 'T1_cut':
                                inp, dec, n, nn, first = compare(orig[(ds, j, L)], log2, c, strict=True)
                                inp_le, dec_le, _, _, _ = compare(orig[(ds, j, L)], log2, c, strict=False)
                                tally(f'{test}_L{L}', ds, inp, dec, n, nn,
                                      {'dec_mismatch_incl_c': int(not dec_le), 'inp_mismatch_incl_c': int(not inp_le),
                                       'errors_after_cut': int(err is not None),
                                       f'cuts_{ctype}': 1, 'future_notes_changed': len(n2) - nk})
                            else:
                                inp, dec, n, nn, first = compare(orig[(ds, j, L)], log2, c, strict=False)
                                tally(f'{test}_L{L}', ds, inp, dec, n, nn,
                                      {'errors_after_cut': int(err is not None), f'cuts_{ctype}': 1,
                                       'notes_replaced': len(n2) - nk,
                                       'replaced_started_before_c': sum(c - L < x[0] <= c for x in n2[nk:])})
                            if err is not None:
                                last_t = log2[-1][0] if log2 else None
                                res.setdefault('errors', []).append({'test': test, 'L': L, 'ds': ds, 'idx': j,
                                                                     'cut': c, 'err': err[:200], 'last_t': last_t})
                            if not (inp and dec):
                                fail(test=test, L=L, ds=ds, idx=j, cut=c, ctype=ctype, inp_same=inp, dec_same=dec,
                                     first=first)
                print(f'{test} L={L} done {time.time() - t_start:.0f}s', flush=True)

        # T3 plan(q) vs unheard pitches
        for L, nq, nq_o in ((0.1, a.nq, a.nq_other), (0.05, a.nq_other, 2), (0.15, a.nq_other, 2)):
            recA.L = L
            for ds, j, s in songs:
                n0 = heard_notes(s)
                r3 = random.Random(f'{SEED}-T3-{L}-{ds}-{j}')
                first_hp = n0[0][0] + L
                qs = [0.5 * x for x in range(int(2 * s['K']) + 1) if first_hp <= 0.5 * x < s['K']]
                fresh = [q for q in qs if any(q - L < n[0] < q + 0.01 for n in n0)]
                other = [q for q in qs if q not in set(fresh)]
                sel = sorted(r3.sample(fresh, min(nq, len(fresh))) + r3.sample(other, min(nq_o, len(other))))
                for q in sel:
                    s2, nchg, ntie = copy_pitch(s, q, L, r3)
                    n2 = heard_notes(s2)
                    if not prefix_ok(n0, n2, q, L, latency=True):
                        tally(f'T3_plan_q_L{L}', ds, True, True, 0, 0, {'construction_mismatch': 1})
                        continue
                    log2, err = run(recA, meta_of(s2), n2, L, stop=q)
                    inp, dec, n, nn, first = compare(orig[(ds, j, L)], log2, q, strict=False)
                    po = [e for e in orig[(ds, j, L)] if e[1] == 0 and e[0] == q]
                    pc = [e for e in log2 if e[1] == 0 and e[0] == q]
                    plan_same = [e[3] for e in po] == [e[3] for e in pc]
                    started = sum(q - L < x[0] <= q + 0.01 for x in n0)
                    tally(f'T3_plan_q_L{L}', ds, inp, dec and plan_same, n, nn,
                          {'plan_q_attack_in_orig': int(bool(po) and po[0][3] is not None),
                           'q_with_note_starting_in_(q-L,q+0.01]': int(started > 0), 'ties_onset_plus_L_eq_q': ntie,
                           'pitches_changed': nchg, 'errors': int(err is not None)})
                    if not (inp and dec and plan_same):
                        fail(test='T3', L=L, ds=ds, idx=j, q=q, inp_same=inp, dec_same=dec, plan_same=plan_same,
                             first=first)
            print(f'T3 L={L} done {time.time() - t_start:.0f}s', flush=True)

        # T4 run history: A again (after everything), fresh B in reverse order
        recA.L = 0.1
        again = {}
        for ds, j, s in songs:
            log, err = run(recA, meta_of(s), heard_notes(s), 0.1)
            again[(ds, j)] = log
        TRIP.phase = 'build'
        objB = B.load(SPEC, train)
        TRIP.phase = None
        recB = Rec(objB, 0.1)
        fresh = {}
        for ds, j, s in reversed(songs):
            log, err = run(recB, meta_of(s), heard_notes(s), 0.1)
            fresh[(ds, j)] = log
        same_again = sum(again[(ds, j)] == orig[(ds, j, 0.1)] for ds, j, _ in songs)
        same_fresh = sum(fresh[(ds, j)] == orig[(ds, j, 0.1)] for ds, j, _ in songs)
        res['tests']['T4_history'] = {'songs': len(songs), 'A_again_identical': same_again,
                                      'B_fresh_reverse_identical': same_fresh}
        print(f'T4 done {time.time() - t_start:.0f}s', flush=True)

        # T6 power: toys
        toy_songs = songs[:2]
        tm = Rec(ToyMemo(), 0.1)
        first_logs = {}
        for ds, j, s in toy_songs:
            first_logs[(ds, j)] = run(tm, meta_of(s), heard_notes(s), 0.1)[0]
        second = {(ds, j): run(tm, meta_of(s), heard_notes(s), 0.1)[0] for ds, j, s in toy_songs}
        ds, j, s = toy_songs[0]
        n0 = heard_notes(s)
        c = cut_times(n0, s['K'], random.Random(1))[1][1]
        s2, _ = copy_cut(s, c, 0.1, random.Random(2), latency=False)
        ToyMemo.mem = {}
        base = run(tm, meta_of(s), n0, 0.1)[0]
        pert = run(tm, meta_of(s2), heard_notes(s2), 0.1)[0]
        t1_toy = compare(base, pert, c, strict=True)[1]
        res['tests']['T6_power'] = {
            'toy_memo_T4_detects': sum(first_logs[k] != second[k] for k in first_logs) == len(first_logs),
            'toy_memo_T1_detects': not t1_toy}
        before = sum(v for k, v in TRIP.hits.items() if k[0] == 'run' and 'causal_probe' in k[2])
        run(Rec(ToyPeek(), 0.1), meta_of(s), n0, 0.1)
        after = sum(v for k, v in TRIP.hits.items() if k[0] == 'run' and 'causal_probe' in k[2])
        res['tests']['T6_power']['toy_peek_trips_T5'] = after > before
        res['tests']['T6_power']['toy_peek_hits'] = after - before
    finally:
        TRIP.uninstall()

    run_hits = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'run'}
    sys_hits = {k: v for k, v in run_hits.items() if 'causal_probe' not in k}
    res['tests']['T5_tripwires'] = {'run_hits_from_system_or_bench': sys_hits,
                                    'run_hits_from_probe_toys': {k: v for k, v in run_hits.items()
                                                                 if 'causal_probe' in k}}
    res['build_reads'] = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'build'}
    res['build_files'] = sorted({os.path.relpath(p, ABS) if p.startswith(ABS) else p for (_, p) in TRIP.paths})
    res['secs'] = round(time.time() - t_start, 1)
    ok = all(t.get('dec_mismatch', 0) == 0 and t.get('inp_mismatch', 0) == 0 and t.get('construction_mismatch', 0) == 0
             for name, t in res['tests'].items() if name.startswith(('T1', 'T2', 'T3')))
    ok = ok and res['tests']['T1_cut_L0.1'].get('dec_mismatch_incl_c', 0) == 0
    ok = ok and res['tests']['T4_history']['A_again_identical'] == len(songs) \
        and res['tests']['T4_history']['B_fresh_reverse_identical'] == len(songs)
    ok = ok and not sys_hits
    res['causal_ok'] = bool(ok)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1, default=str)
    for name, t in res['tests'].items():
        print(name, json.dumps(t, default=str)[:600], flush=True)
    print('build files:', res['build_files'], flush=True)
    print('build reads:', json.dumps(res['build_reads'])[:1500], flush=True)
    print('errors:', len(res.get('errors', [])), (res.get('errors') or [])[:3], flush=True)
    print('fail examples:', res['fail_examples'][:5], flush=True)
    print(f"causal_ok {res['causal_ok']}  ({res['secs']} s)  -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
