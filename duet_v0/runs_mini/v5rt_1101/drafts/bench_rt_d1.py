"""bench_rt: latency-aware real-time scoring for Duet (run v5rt_1101). Val split only: this bundle has no test split.

SPEC (run v5rt_1101, phase 1; this file implements it and is frozen after phase 1, see runs_mini/v5rt_1101/BENCH_SHA.txt)
- Data and scoring base = bench3: songs = bench3.eval_sets('val') (bach 17, folk 170, pop 300, pop909 40). Her HEARD
  melody = bench3's own heard melody (B2.heard: 10 % mishearing, seed crc32(id) ^ 5160); --seed2 uses the alternative
  seed exactly as futures/v4_selfiter/seed2.py does. Systems only ever receive heard pitches. Scoring uses TRUE pitches
  and TRUE chords exactly like futures/v3_nolatency/rescore_nd.py's music_nd: only her singing time from beat J = 4,
  only labelled beats (z >= 0); a moment is a hit when Duet sounds h >= 0 with |m - h| mod 12 in theory.CONSONANT minus
  {0} and (h - tonic) mod 12 in CT[z]; Duet silence is a miss. Dataset score = hit time / counted time over its songs;
  headline = mean of the 4 dataset scores. Integrate exactly (piecewise), not by sampling.
- Time is in beats. Nominal tempo 100 bpm, 1 beat = 600 ms. L = hearing latency in beats: default 0.1 (60 ms); also
  0.05 and 0.15.
- Duet's sounding note is a step function set by decision points processed in time order (ties: plan before hear):
  (a) plan points at every multiple of 0.5 beat, q = 0, 0.5, 1, ... < K: obj.plan(q, heard) returns the note Duet
      sounds from time q on; None = keep the sounding note; -1 = silence. heard = her notes already HEARD at time q,
      i.e. with onset + L <= q, as a list of [onset, offset, heard_midi] in beats, where offset is given only if
      offset + L <= q, else None. A note that started less than L before q is NOT in heard.
  (b) hear points at onset_i + L for each of her notes i (sorted by onset): obj.hear(i, onset_i, heard_midi_i,
      prev_dur) with bench3 step's arguments (prev_dur = her previous note's duration, None for the first) returns the
      note from onset_i + L on; None = keep; -1 = silence.
  Before the first decision Duet is silent.
- System files: futures/v5_rt/rt_systems/<name>.py with build(train) -> obj having start(meta) (meta exactly as
  bench3's: tonic, M, K, pos), plan(q, heard), hear(i, onset, midi, prev_dur). Systems never read song data, true
  pitches, chords or os.environ; they learn only from train = bench3.train_sets() or files they trained from it.
- Adapters: 'reactive:<v3 system>' = plan always None, hear = that bench3 system's step(); 'oracle:<v3 system>' = at a
  plan point q where one of her notes starts exactly at q, plan is told that note's heard pitch and calls step() then
  (a ceiling: perfect knowledge of her next note, impossible live), and hear() returns None for that note; all other
  notes as reactive.
- Reported per run (per dataset and headline): rt_nd; harsh = share of counted time with h >= 0 and |m - h| mod 12 in
  {1, 2, 6, 10, 11} against her TRUE note; double = share with interval class 0; nd_first and harsh_first = the same
  restricted to the first L of each of her notes; on_time_right = share of her notes (onset >= J) whose sounding note
  just after onset equals the one just after onset + L and is a hit at the onset; changes = share of her notes where
  the sounding note just after onset + L differs from just after onset and both are >= 0; late_entry = share where
  Duet is silent just after onset and sounds after onset + L; mean and max us per decision call.
- CLI: .venv/bin/python futures/v5_rt/bench_rt.py <spec> [--L 0.1] [--seed2] --out <new json path>;
  <spec> = reactive:<name> | oracle:<name> | rt:<name>; prints one summary line; writes its JSON with mode 'x' under
  runs_mini/v5rt_1101/scores/.
- KEEP RULE for a candidate, vs reactive:r2_mix_v2 at L = 0.1, same seed: headline rt_nd >= reactive + 0.005; harsh
  headline <= reactive's harsh and no dataset's harsh above reactive's + 0.005; no dataset's rt_nd more than 0.01
  below reactive's; seed2 headline delta (candidate seed2 minus reactive seed2) >= +0.003; mean us per decision < 2000.

How this file reads the SPEC where it leaves room (decided in phase 1, frozen with the file):
- heard is a fresh list on every plan call; each item is a tuple (onset, offset or None, heard_midi), so a system
  cannot change the bench's copy. Times are compared exactly as floats, a hear point at onset + L (the same float in
  heard, in the event order and in the scoring). In val no onset + L falls exactly on a plan point for L = 0.05, 0.1,
  0.15; at L = 0 every hear point equals its onset, and a plan point at a note's onset comes first and already lists
  that note (onset + 0 <= q) before its hear() runs.
- "Just after x" = Duet's note once every decision at a time <= x has been applied (several decisions at one time:
  the last one processed wins; a plan point's note overridden by a hear at the same time lasts zero time).
- Per-note shares (on_time_right, changes, late_entry) are over her notes with onset >= J whose onset beat int(onset)
  is labelled (z >= 0): a hit is only defined on labelled beats; "a hit at the onset" uses that beat's true chord and
  her true pitch. A note shorter than L is still judged at onset + L (after it ended).
- nd_first / harsh_first count the counted time inside [onset, onset + L) of each of her notes (cut at its offset);
  they are None when there is no such time (L = 0).
- Headline of every share = mean of the 4 dataset values (None if one is None). us_mean / us_max (headline and per
  dataset) are over every plan() and hear() call; us_mean_plan / us_mean_hear split them. Only the system's call is
  timed, not the bench's bookkeeping.
- Oracle: "starts exactly at q" is exact float equality onset == q. pop909 onsets lie 0.0002-0.0003 beat after the
  grid (performance timing), so only about 4 % of pop909 notes start on a plan point (bach 99.7 %, folk 82 %, pop 86 %):
  the oracle is close to reactive on pop909 and is a weak ceiling there. step() must see her notes in order, so if an
  earlier note's hear point has not been reached at q the note at q stays reactive (counted as oracle_blocked; each
  row gives oracle_notes and oracle_blocked).
- At L = 0, reactive:<v3> reproduces rescore_nd's music_nd and double exactly (checked in phase 1; her val notes never
  overlap, so one sounding note at a time is what bench3 scores).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python futures/v5_rt/bench_rt.py reactive:r2_mix_v2 --L 0.1 \
      --out runs_mini/v5rt_1101/scores/<new name>.json
"""
import argparse
import bisect
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import zlib

ABS = '/Users/san/claude_sandbox/music/duet_v0'
V3 = os.path.join(ABS, 'futures', 'v3_nolatency')
V5 = os.path.join(ABS, 'futures', 'v5_rt')
SCORES = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'scores')
sys.path.insert(0, V3)                     # bench3 and the v3 systems package `systems`
sys.path.insert(0, V5)                     # the rt_systems package
import bench3  # noqa: E402  (also puts duet_beat on sys.path)
from theory import CONSONANT, CT  # noqa: E402

J = bench3.J                               # her singing time counts from beat 4
HARSH = bench3.HARSH                       # 2nds, tritone, 7ths
GOOD = CONSONANT - {0}                     # rescore_nd's music_nd: consonant and not her unison/octave
STEP = 0.5                                 # plan points every half beat
SHARES = ('rt_nd', 'harsh', 'double', 'nd_first', 'harsh_first', 'on_time_right', 'changes', 'late_entry')


def heard_seed2(song, p=bench3.B2.NOISE):
    """futures/v4_selfiter/seed2.py's heard(): bench.heard with the seed ^ 5161 instead of ^ 5160 (copied, because
    seed2.py runs when imported)."""
    rng = random.Random(zlib.crc32(str(song.get('id', song.get('title', ''))).encode()) ^ 5161)
    sop = []
    for on, off, m in song['sop']:
        if rng.random() < p:
            pc = (m + rng.randint(1, 11)) % 12           # one of the other 11 pitch classes
            m = min((m + d for d in range(-6, 7) if (m + d) % 12 == pc), key=lambda x: abs(x - m))
        sop.append([on, off, m])
    beats = [{'pos': b['pos'], 'z': -1, 'onset_pcs': []} for b in song['beats']]
    return dict(song, sop=sop, beats=beats)


# ---------------------------------------------------------------------------------------------------- adapters
class Reactive:
    """'reactive:<v3 system>': plan() never changes Duet's note; hear() is the bench3 system's step()."""

    def __init__(self, v3):
        self.v3 = v3

    def start(self, meta):
        self.v3.start(meta)

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return int(self.v3.step(i, onset, midi, prev_dur))


class Oracle(Reactive):
    """'oracle:<v3 system>': a note that starts exactly on a plan point q is answered at q by step() with its heard
    pitch (impossible live: a ceiling); its hear() then keeps that note. Only the bench calls tell()."""

    def __init__(self, v3):
        super().__init__(v3)
        self.n_early = self.n_blocked = 0                # running counts over all songs

    def tell(self, notes):
        self.notes = notes                               # this song's heard notes (onset, offset, midi), by onset

    def start(self, meta):
        super().start(meta)
        self.next = 0                                    # the note step() must see next: always in note order
        self.p = 0                                       # first note that does not start before the last plan point
        self.early = set()

    def plan(self, q, heard):
        out, ns = None, self.notes
        while self.p < len(ns) and ns[self.p][0] < q:
            self.p += 1
        j = self.p
        while j < len(ns) and ns[j][0] == q:
            if j != self.next:                           # an earlier note is not heard yet: stay reactive
                self.n_blocked += 1
                break
            prev = None if j == 0 else float(ns[j - 1][1] - ns[j - 1][0])
            out = self._step(j, ns[j][0], ns[j][2], prev)
            self.early.add(j)
            self.n_early += 1
            j += 1
        return out

    def hear(self, i, onset, midi, prev_dur):
        return None if i in self.early else self._step(i, onset, midi, prev_dur)

    def _step(self, i, onset, midi, prev_dur):
        assert i == self.next, (i, self.next)
        self.next = i + 1
        return int(self.v3.step(i, float(onset), int(midi), prev_dur))


def load(spec, train):
    kind, _, name = spec.partition(':')
    if kind == 'rt' and name:
        return importlib.import_module('rt_systems.' + name).build(train)
    if kind in ('reactive', 'oracle') and name:
        v3 = importlib.import_module('systems.' + name).build(train)
        return Reactive(v3) if kind == 'reactive' else Oracle(v3)
    raise SystemExit(f'bad spec {spec!r}: use reactive:<v3 system> | oracle:<v3 system> | rt:<rt system>')


# ---------------------------------------------------------------------------------------------------- one song
def simulate(obj, meta, notes, L):
    """Run one song through obj. notes = her HEARD notes (onset, offset, midi) sorted by onset. Returns (T, H, us_plan,
    us_hear): Duet sounds H[j] from time T[j] on (T strictly increasing; silent before T[0]); us_* = microseconds of
    each plan() / hear() call."""
    obj.start(meta)
    K = meta['K']
    ev = [(STEP * j, 0, j) for j in range(int(math.ceil(K / STEP)) + 1) if STEP * j < K]     # plan points
    ev += [(on + L, 1, i) for i, (on, off, m) in enumerate(notes)]                            # hear points
    ev.sort()                                            # time order; ties: plan (0) before hear (1)
    heard, opened, nxt = [], [], 0
    T, H, us_plan, us_hear = [], [], [], []
    for t, kind, j in ev:
        if kind == 0:
            while nxt < len(notes) and notes[nxt][0] + L <= t:
                heard.append((notes[nxt][0], None, notes[nxt][2]))
                opened.append(nxt)
                nxt += 1
            still = []
            for a in opened:
                if notes[a][1] + L <= t:
                    heard[a] = notes[a]
                else:
                    still.append(a)
            opened = still
            view = list(heard)
            c0 = time.perf_counter()
            h = obj.plan(t, view)
            us_plan.append(1e6 * (time.perf_counter() - c0))
        else:
            on, off, m = notes[j]
            prev = None if j == 0 else float(notes[j - 1][1] - notes[j - 1][0])
            c0 = time.perf_counter()
            h = obj.hear(j, on, m, prev)
            us_hear.append(1e6 * (time.perf_counter() - c0))
        if h is not None:
            h = int(h)
            if T and T[-1] == t:
                H[-1] = h
            else:
                T.append(t)
                H.append(h)
    return T, H, us_plan, us_hear


def hit(h, m, t, z):
    """rescore_nd's music_nd moment: Duet sounds h >= 0, consonant with her true note m but not her unison/octave, and a
    tone of the true chord z >= 0 in key t."""
    return h >= 0 and abs(m - h) % 12 in GOOD and (h - t) % 12 in CT[z]


def at(T, H, x):
    """Duet's note just after time x; -1 = silent."""
    j = bisect.bisect_right(T, x) - 1
    return H[j] if j >= 0 else -1


def pieces(T, H, a, b):
    """Duet's step function on [a, b] as (x0, x1, h) pieces in time order."""
    j = bisect.bisect_right(T, a) - 1
    x0 = a
    while True:
        x1 = T[j + 1] if j + 1 < len(T) and T[j + 1] < b else b
        yield x0, x1, (H[j] if j >= 0 else -1)
        if x1 >= b:
            return
        x0, j = x1, j + 1


def score_song(song, T, H, L, acc):
    """Add one song's exact integrals against her TRUE notes and chords to acc (rescore_nd's loop, split at Duet's
    changes)."""
    true = sorted(song['sop'], key=lambda n: n[0])
    z = [b['z'] for b in song['beats']]
    t = song['tonic']
    for on, off, m in true:
        k0 = int(on)
        if on >= J and k0 < len(z) and z[k0] >= 0:
            v0, v1 = at(T, H, on), at(T, H, on + L)
            acc['notes'] += 1
            acc['on_time_right'] += v0 == v1 and hit(v0, m, t, z[k0])
            acc['changes'] += v0 != v1 and v0 >= 0 and v1 >= 0
            acc['late_entry'] += v0 < 0 and v1 >= 0
        e = on + L                                       # end of her note's first L
        for k in range(max(J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            acc['d'] += ov
            for x0, x1, h in pieces(T, H, max(on, k), min(off, k + 1)):
                w, f = x1 - x0, min(x1, e) - x0
                g = hit(h, m, t, z[k])
                bad = h >= 0 and abs(m - h) % 12 in HARSH
                acc['hit'] += w * g
                acc['harsh'] += w * bad
                acc['double'] += w * (h >= 0 and abs(m - h) % 12 == 0)
                if f > 0:
                    acc['d_first'] += f
                    acc['hit_first'] += f * g
                    acc['harsh_first'] += f * bad


# ---------------------------------------------------------------------------------------------------- a run
def _ratio(a, b):
    return a / b if b > 0 else None


def row(acc, us_plan, us_hear, n_songs):
    us = us_plan + us_hear
    return {'rt_nd': _ratio(acc['hit'], acc['d']), 'harsh': _ratio(acc['harsh'], acc['d']),
            'double': _ratio(acc['double'], acc['d']),
            'nd_first': _ratio(acc['hit_first'], acc['d_first']),
            'harsh_first': _ratio(acc['harsh_first'], acc['d_first']),
            'on_time_right': _ratio(acc['on_time_right'], acc['notes']),
            'changes': _ratio(acc['changes'], acc['notes']), 'late_entry': _ratio(acc['late_entry'], acc['notes']),
            'us_mean': _ratio(sum(us), len(us)), 'us_max': max(us) if us else None,
            'us_mean_plan': _ratio(sum(us_plan), len(us_plan)), 'us_mean_hear': _ratio(sum(us_hear), len(us_hear)),
            'us_total': sum(us), 'calls_plan': len(us_plan), 'calls_hear': len(us_hear), 'songs': n_songs,
            'counted_beats': acc['d'], 'first_beats': acc['d_first'], 'notes': int(acc['notes'])}


def evaluate(spec, L=0.1, seed2=False):
    heard = bench3.B2.heard
    if seed2:
        bench3.B2.heard = heard = heard_seed2          # as seed2.py does, before the system is built
    obj = load(spec, bench3.train_sets())
    rows = {}
    for ds, songs in bench3.eval_sets('val').items():
        acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                             'on_time_right', 'changes', 'late_entry'), 0.0)
        up, uh = [], []
        oracle = isinstance(obj, Oracle)
        if oracle:
            e0, b0 = obj.n_early, obj.n_blocked
        for s in songs:
            notes = [(float(on), float(off), int(m)) for on, off, m in sorted(heard(s)['sop'], key=lambda n: n[0])]
            meta = {'tonic': s['tonic'], 'M': s['M'], 'K': s['K'], 'pos': [b['pos'] for b in s['beats']]}
            if oracle:
                obj.tell(notes)
            T, H, a, b = simulate(obj, meta, notes, L)
            up += a
            uh += b
            score_song(s, T, H, L, acc)
        rows[ds] = row(acc, up, uh, len(songs))
        if oracle:
            rows[ds]['oracle_notes'], rows[ds]['oracle_blocked'] = obj.n_early - e0, obj.n_blocked - b0
    assert list(rows) == ['bach', 'folk', 'pop', 'pop909'], list(rows)
    head = {k: (sum(r[k] for r in rows.values()) / len(rows) if all(r[k] is not None for r in rows.values()) else None)
            for k in SHARES}
    calls = sum(r['calls_plan'] + r['calls_hear'] for r in rows.values())
    head['us_mean'] = _ratio(sum(r['us_total'] for r in rows.values()), calls)
    head['us_max'] = max(r['us_max'] for r in rows.values() if r['us_max'] is not None)
    head['us_mean_plan'] = _ratio(sum(r['us_mean_plan'] * r['calls_plan'] for r in rows.values() if r['calls_plan']),
                                  sum(r['calls_plan'] for r in rows.values()))
    head['us_mean_hear'] = _ratio(sum(r['us_mean_hear'] * r['calls_hear'] for r in rows.values() if r['calls_hear']),
                                  sum(r['calls_hear'] for r in rows.values()))
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'spec': spec, 'L': L, 'seed': 'seed2' if seed2 else 'seed1', 'split': 'val', 'headline': head, 'rows': rows,
            'bench': os.path.abspath(__file__), 'bench_sha256': sha, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def line(out):
    h = out['headline']
    f = lambda x: 'n/a' if x is None else f'{x:.4f}'  # noqa: E731
    return (f"{out['spec']} L={out['L']:g} {out['seed']}: rt_nd {f(h['rt_nd'])} harsh {f(h['harsh'])} "
            f"double {f(h['double'])} nd_first {f(h['nd_first'])} harsh_first {f(h['harsh_first'])} "
            f"on_time_right {f(h['on_time_right'])} changes {f(h['changes'])} late_entry {f(h['late_entry'])} | "
            + ' '.join(f"{ds} {f(r['rt_nd'])}" for ds, r in out['rows'].items())
            + f" | us/decision mean {h['us_mean']:.0f} max {h['us_max']:.0f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='bench_rt: latency-aware real-time scoring (the SPEC is the docstring)')
    ap.add_argument('spec', help='reactive:<v3 system> | oracle:<v3 system> | rt:<rt system>')
    ap.add_argument('--L', type=float, default=0.1, help='hearing latency in beats (0.1 = 60 ms at 100 bpm)')
    ap.add_argument('--seed2', action='store_true', help="her mishearing drawn with seed2.py's seed (^ 5161)")
    ap.add_argument('--out', required=True, help='new .json file under runs_mini/v5rt_1101/scores/')
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(SCORES + os.sep) or not path.endswith('.json'):
        sys.exit(f'--out must be a new .json file under {SCORES}/')
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    if not a.L >= 0:
        sys.exit('--L must be >= 0')
    res = evaluate(a.spec, a.L, a.seed2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(line(res), flush=True)
