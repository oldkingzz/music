"""v5rt_1101 phase 1: decision order, heard contents and exact integration of a bench_rt file on tiny hand-made songs
(no val data, no audio). Prints PASS/FAIL per check and exits 1 on any FAIL.

  python check_tiny_v1.py <path of the bench_rt file under test>
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('bench_under_test', sys.argv[1])
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

FAILS = []


def check(name, got, want, tol=1e-12):
    if isinstance(want, float) and got is not None:
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'   got {got!r}   want {want!r}'))
    if not ok:
        FAILS.append(name)


class Probe:
    """Logs every call; returns scripted notes (plan by q, hear by note index; missing = None)."""

    def __init__(self, plan_out, hear_out, mutate=False):
        self.po, self.ho, self.mutate, self.log = plan_out, hear_out, mutate, []

    def start(self, meta):
        self.log.append(('start', meta['K']))

    def plan(self, q, heard):
        self.log.append(('plan', q, list(heard)))
        if self.mutate:                                  # a careless system: must not change the bench's state
            heard.append((99.0, None, 0))
        return self.po.get(q)

    def hear(self, i, onset, midi, prev_dur):
        self.log.append(('hear', i, onset, midi, prev_dur))
        return self.ho.get(i)


class FakeV3:
    """A bench3-style system: step() returns 60 + i and records its calls."""

    def start(self, meta):
        self.calls = []

    def step(self, i, onset, midi, prev_dur):
        self.calls.append((i, onset, midi, prev_dur))
        return 60 + i


def plan_log(p, q):
    return [c[2] for c in p.log if c[0] == 'plan' and c[1] == q][0]


def score(song, T, H, L):
    acc = dict.fromkeys(('d', 'hit', 'harsh', 'double', 'd_first', 'hit_first', 'harsh_first', 'notes',
                         'on_time_right', 'changes', 'late_entry'), 0.0)
    B.score_song(song, T, H, L, acc)
    return acc, B.row(acc, [1.0], [1.0], 1)


def rescore_nd_ref(song, hs):
    """rescore_nd.py's loop, verbatim logic, for one song and Duet notes hs (one per her note)."""
    from theory import CONSONANT, CT
    num = den = dbl = 0.0
    true = sorted(song['sop'], key=lambda n: n[0]); z = [b['z'] for b in song['beats']]; t = song['tonic']  # noqa: E702
    for i, (on, off, m) in enumerate(true):
        h = hs[i]
        for k in range(max(B.J, int(on)), int(off) + 1):
            ov = min(off, k + 1) - max(on, k)
            if ov <= 1e-9 or k >= len(z) or z[k] < 0:
                continue
            den += ov
            if h >= 0:
                iv = abs(m - h) % 12
                num += ov * (iv in CONSONANT and iv != 0 and (h - t) % 12 in CT[z[k]])
                dbl += ov * (iv == 0)
    return num / den, dbl / den


# Song A: two notes in C major; true chord I (C E G) on beats 0-4, V (G B D) on beat 5, beats 6-7 unlabelled.
A = {'id': 'tinyA', 'tonic': 0, 'M': 4, 'K': 8,
     'beats': [{'pos': k % 4, 'z': 0 if k <= 4 else (4 if k == 5 else -1)} for k in range(8)],
     'sop': [[4.0, 4.4, 72], [4.4, 5.25, 76]]}                    # her TRUE notes: C5, then E5
NA = [(4.0, 4.4, 72), (4.4, 5.25, 77)]                            # HEARD: the E5 is misheard as F5
META = {'tonic': 0, 'M': 4, 'K': 8, 'pos': [k % 4 for k in range(8)]}

print('--- song A, L = 0.1: plan 4.5 -> B4 then hear(1) -> C4 at the same time (tie), plan 5.0 -> D5, plan 5.5 -> silence')
check('4.4 + 0.1 is exactly the plan point 4.5 (a true tie)', 4.4 + 0.1 == 4.5, True)
p = Probe({4.5: 71, 5.0: 74, 5.5: -1}, {0: 60, 1: 60}, mutate=True)
T, H, up, uh = B.simulate(p, META, NA, 0.1)
order = [(c[0], c[1]) for c in p.log[1:]]
want = [('plan', 0.5 * j) for j in range(9)] + [('hear', 0), ('plan', 4.5), ('hear', 1)] + \
       [('plan', 0.5 * j) for j in range(10, 16)]
check('call order: plans 0..4.0, hear(0) at 4.1, plan 4.5 BEFORE hear(1) at 4.5, plans 5.0..7.5', order, want)
check('16 plan calls and 2 hear calls timed', (len(up), len(uh)), (16, 2))
check('heard at plan 4.0 is empty (her first note is heard only at 4.1)', plan_log(p, 4.0), [])
check('heard at plan 4.5: note 0 with its offset (4.4 + L <= 4.5), note 1 (heard F5 = 77) without offset',
      plan_log(p, 4.5), [(4.0, 4.4, 72), (4.4, None, 77)])
check('heard at plan 5.0: note 1 still without offset (5.25 + L > 5.0); a system appending to heard changed nothing',
      plan_log(p, 5.0), [(4.0, 4.4, 72), (4.4, None, 77)])
check('heard at plan 5.5: note 1 offset given (5.35 <= 5.5)', plan_log(p, 5.5), [(4.0, 4.4, 72), (4.4, 5.25, 77)])
hears = [c for c in p.log if c[0] == 'hear']
check('hear(0) gets bench3 step args (0, 4.0, 72, None)', hears[0][1:], (0, 4.0, 72, None))
check('hear(1) gets the HEARD pitch 77 and prev_dur = 4.4 - 4.0', hears[1][1:], (1, 4.4, 77, 4.4 - 4.0))
check('step function: silent < 4.1, C4 from 4.1, C4 from 4.5 (hear overrode the plan at the tie), D5 from 5.0, '
      'silent from 5.5', (T, H), ([4.0 + 0.1, 4.5, 5.0, 5.5], [60, 60, 74, -1]))
acc, r = score(A, T, H, 0.1)
check('counted time 1.25 beats (beats 4 and 5; beat 5 has chord V)', acc['d'], 1.25)
check('rt_nd = 0.6 / 1.25 (only C4 under E5 on [4.4, 5.0] is a hit; silence and C4 under C5 are misses)', r['rt_nd'], 0.48)
check('harsh = 0.25 / 1.25 (D5 under E5 on [5.0, 5.25])', r['harsh'], 0.2)
check('double = 0.3 / 1.25 (C4 under C5 on [4.1, 4.4])', r['double'], 0.24)
check('first-L time 0.2 beat', acc['d_first'], 0.2)
check('nd_first = 0.1 / 0.2 (note 0 starts silent, note 1 starts on the kept C4)', r['nd_first'], 0.5)
check('harsh_first = 0', r['harsh_first'], 0.0)
check('per-note set: 2 notes', r['notes'], 2)
check('on_time_right = 1/2 (note 1: C4 just after 4.4 and after 4.5, a hit)', r['on_time_right'], 0.5)
check('changes = 0', r['changes'], 0.0)
check('late_entry = 1/2 (note 0: silent at 4.0, C4 after 4.1)', r['late_entry'], 0.5)

print('--- song A, L = 0.1: plan 4.0 -> B4 (wrong start), hear(0) -> E4 (correction), hear(1) keeps E4')
p = Probe({4.0: 71}, {0: 64})
T, H, up, uh = B.simulate(p, META, NA, 0.1)
check('step function: B4 from 4.0, E4 from 4.1', (T, H), ([4.0, 4.0 + 0.1], [71, 64]))
acc, r = score(A, T, H, 0.1)
check('rt_nd = 0.3 / 1.25 (E4 under C5 on [4.1, 4.4])', r['rt_nd'], 0.24)
check('harsh = 0.1 / 1.25 (B4 under C5 on [4.0, 4.1])', r['harsh'], 0.08)
check('double = 0.85 / 1.25 (E4 under E5)', r['double'], 0.68)
check('harsh_first = 0.1 / 0.2', r['harsh_first'], 0.5)
check('nd_first = 0', r['nd_first'], 0.0)
check('changes = 1/2 (note 0: B4 -> E4)', r['changes'], 0.5)
check('on_time_right = 0 (note 1 kept E4, but it doubles her E5)', r['on_time_right'], 0.0)
check('late_entry = 0', r['late_entry'], 0.0)

print('--- song A, L = 0: a plan point at an onset runs first and already lists that note; result = rescore_nd')
p = Probe({}, {0: 64, 1: 74})
T, H, up, uh = B.simulate(p, META, NA, 0.0)
order = [(c[0], c[1]) for c in p.log[1:]]
want = [('plan', 0.5 * j) for j in range(9)] + [('hear', 0), ('hear', 1)] + [('plan', 0.5 * j) for j in range(9, 16)]
check('call order at L = 0: plan 4.0 then hear(0) at 4.0, hear(1) at 4.4, plan 4.5', order, want)
check('heard at plan 4.0 at L = 0: note 0 without offset', plan_log(p, 4.0), [(4.0, None, 72)])
check('step function', (T, H), ([4.0, 4.4], [64, 74]))
acc, r = score(A, T, H, 0.0)
nd, dbl = rescore_nd_ref(A, [64, 74])
check('rt_nd equals rescore_nd music_nd (0.4 / 1.25)', r['rt_nd'], nd, tol=0.0)
check('double equals rescore_nd double', r['double'], dbl, tol=0.0)
check('harsh = 0.85 / 1.25 (D5 under E5)', r['harsh'], 0.68)
check('nd_first is None at L = 0', r['nd_first'], None)
check('on_time_right = 1/2 at L = 0 (note 0: E4 under C5 on chord I)', r['on_time_right'], 0.5)

print('--- song B, oracle: note 1 starts 0.05 before the plan point 4.5 where note 2 starts')
NB = [(4.0, 4.45, 72), (4.45, 4.5, 74), (4.5, 5.0, 76), (5.25, 6.0, 72)]
fake = FakeV3()
o = B.Oracle(fake)
o.tell(NB)
T, H, up, uh = B.simulate(o, META, NB, 0.1)
check('oracle L=0.1: step() in note order', [c[0] for c in fake.calls], [0, 1, 2, 3])
check('oracle L=0.1: note 0 answered at plan 4.0; note 2 blocked (note 1 unheard at 4.5) so reactive at 4.6',
      (T, H), ([4.0, 4.45 + 0.1, 4.5 + 0.1, 5.25 + 0.1], [60, 61, 62, 63]))
check('oracle L=0.1: counts early 1, blocked 1', (o.n_early, o.n_blocked), (1, 1))
check('oracle step args for note 0 at the plan point', fake.calls[0], (0, 4.0, 72, None))
fr = FakeV3()
T2, H2, _, _ = B.simulate(B.Reactive(fr), META, NB, 0.1)
check('reactive L=0.1: every note answered at onset + L', (T2, H2), ([4.0 + 0.1, 4.45 + 0.1, 4.5 + 0.1, 5.25 + 0.1],
                                                                      [60, 61, 62, 63]))
fake = FakeV3()
o = B.Oracle(fake)
o.tell(NB)
T, H, _, _ = B.simulate(o, META, NB, 0.0)
fr = FakeV3()
T2, H2, _, _ = B.simulate(B.Reactive(fr), META, NB, 0.0)
check('oracle L=0 equals reactive L=0 (same step function)', (T, H), (T2, H2))
check('oracle L=0 equals reactive L=0 (same step() calls)', fake.calls, fr.calls)
check('oracle L=0: counts early 2 (notes 0 and 2), blocked 0', (o.n_early, o.n_blocked), (2, 0))

print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
