"""Rehearsal step: analyse a reference recording once and cache everything the live engine needs.

    reference.wav --> features F (FeatureTracker, the same causal features the follower sees live)
                  --> notes (offline_notes, detector delay removed)
                  --> harmony plan (plan_harmony, or a JSON plan from the harmony model)
                  --> breaths (silent gaps), join point, beat length
                  --> optional timing profile learned from a rehearsal take of the singer

The cache is two files: <prefix>.npz (arrays) and <prefix>.json (everything else, readable).
find_gaps / match / calibrate are copied from run_v0.py (unchanged logic) so the live app does
not break if the offline simulator is edited.
"""
import hashlib
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np

import singer
from dsp import HOP, SR
from follow import Accompanist, FeatureTracker, Follower, analyse, offline_notes, plan_harmony, profile_from_anchors

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parent
HARMONY_MODEL = ROOT / 'harmony' / 'model.py'
CACHE_VERSION = 1
CAL_PATH = LIVE / 'cache' / 'calibration.json'
EVAL_DIR = ROOT / 'eval'
FROZEN_V2 = EVAL_DIR / 'results' / 'v2_frozen.json'
BIAS_KINDS = ('attack', 'glide', 'repeat')


# ---------------------------------------------------------------- the evaluation's front end (optional)
def eval_features():
    """eval/duet_eval/features.py: the evaluation's input level gate (frame_db, gate_threshold,
    apply_gate, StreamGate). Imported, not copied, so the live gate IS the evaluated gate."""
    if str(EVAL_DIR) not in sys.path:
        sys.path.append(str(EVAL_DIR))
    from duet_eval import features
    return features


def gate_from_reference(x_ref, gate_db):
    """Gate settings fixed at rehearsal time, as in the evaluation (features.gate_threshold):
    threshold = 90th percentile of the reference's 10 ms frame levels - gate_db (dBFS)."""
    f = eval_features()
    return {'gate_db': float(gate_db), 'level_p90_db': float(np.percentile(f.frame_db(x_ref), 90)),
            'thr_db': float(f.gate_threshold(x_ref, gate_db)), 'source': 'eval/duet_eval/features.py'}


def gate_audio(x, gate):
    """Offline gate (features.apply_gate) for the reference / rehearsal take; x unchanged if gate is None."""
    return x if gate is None else eval_features().apply_gate(x, gate['thr_db'])


def parse_bias(spec):
    """--bias SPEC -> ({kind: seconds}, source).
    SPEC = 'attack,glide,repeat' in seconds (e.g. '-0.023,-0.042,-0.056'), or the name of a dataset
    in eval/results/v2_frozen.json front.bias ('dcs', 'ecd'): the evaluation's per-dataset detector
    bias (live anchor time - annotated onset, calibrated on the tune split)."""
    spec = str(spec).strip()
    if ',' in spec:
        vals = [float(v) for v in spec.split(',')]
        if len(vals) != len(BIAS_KINDS):
            raise ValueError(f'--bias needs {len(BIAS_KINDS)} values ({",".join(BIAS_KINDS)}), got {spec!r}')
        return dict(zip(BIAS_KINDS, vals)), 'command line (attack,glide,repeat)'
    table = json.loads(FROZEN_V2.read_text())['front']['bias']
    if spec not in table:
        raise ValueError(f'--bias {spec!r}: not "a,g,r" and not a dataset in {FROZEN_V2.name} ({", ".join(table)})')
    return {k: float(v) for k, v in table[spec].items()}, f'eval/results/v2_frozen.json front.bias.{spec}'


# ---------------------------------------------------------------- audio input
def load_audio(path):
    """Any soundfile-readable file -> mono float32 at 24 kHz (polyphase resampling if needed)."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x.astype(np.float32)


# ---------------------------------------------------------------- copied from run_v0.py
def find_gaps(F, min_len=0.12):
    gaps, t, v = [], F['t'], F['voiced']
    i = 0
    while i < len(v):
        if not v[i]:
            s = i
            while i < len(v) and not v[i]:
                i += 1
            if s > 0 and i < len(v) and t[i - 1] - t[s] >= min_len:
                gaps.append((float(t[s]), float(t[i - 1])))
        else:
            i += 1
    return gaps


def match(times, truth, tol=0.15):
    truth = np.asarray(truth)
    out = []
    for x in times:
        k = int(np.argmin(np.abs(truth - x)))
        out.append(k if abs(truth[k] - x) <= tol else None)
    return out


def _follow(y, F, acc):
    ft, fol = FeatureTracker(), Follower(F)
    for k in range(len(y) // HOP):
        ft.step(y[k * HOP:(k + 1) * HOP])
        fol.step(ft.m[-1], ft.v[-1], ft.nov[-1])
        acc.step((k + 1) * HOP / SR, ft, fol, 0.0)


def calibrate_detectors(force=False):
    """Detector delays per onset kind, measured on the synthetic singer (exact onsets known).
    Same procedure as run_v0.calibrate(); cached in live/cache/calibration.json."""
    if CAL_PATH.exists() and not force:
        return json.loads(CAL_PATH.read_text())
    xs, true_notes = singer.render(singer.score_notes())
    F = analyse(xs)
    det = offline_notes(F)
    breaths = [(a + b) / 2 for a, b in find_gaps(F)]
    true_on = [n['on'] for n in true_notes]
    idx = match([n['on'] for n in det], true_on)
    ref_bias = {}
    for kind in ('attack', 'glide', 'repeat'):
        d = [nt['on'] - true_on[k] for nt, k in zip(det, idx) if k is not None and true_notes[k]['kind'] == kind]
        ref_bias[kind] = float(np.median(d)) if d else 0.0
    cal = [dict(nt, on=nt['on'] - ref_bias.get(nt['kind'], 0.0)) for nt in det]
    acc = Accompanist(cal, [], 0.0, breaths=breaths)
    _follow(xs, F, acc)
    by_r = {nt['on']: k for k, nt in enumerate(cal)}
    live_bias = {}
    for kind in ('attack', 'glide', 'repeat'):
        d = [t - true_on[idx[by_r[r]]] for t, r in acc.anchors
             if idx[by_r[r]] is not None and cal[by_r[r]]['kind'] == kind]
        live_bias[kind] = float(np.median(d)) if d else 0.0
    out = {'ref_bias': ref_bias, 'live_bias': live_bias, 'source': 'singer.render(singer.score_notes())',
           'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- beat length
def estimate_beat(notes, breaths, lo=0.30, hi=1.25, tol=0.08):
    """Beat length (reference seconds) from the inter-onset intervals inside phrases.

    For each candidate period P, score how many IOIs sit close to an allowed multiple of P
    (1, 1.5 (dotted), 2, 3, 4 ... beats); among the local maxima of that score that explain
    >= 70 % as many IOIs as the best one, take the LONGEST P (a half-beat grid always fits as
    well, so the longest good grid is the beat). Heuristic; override with --beat.
    """
    ons = [n['on'] for n in notes]
    ph = [sum(1 for g in breaths if g < r) for r in ons]
    ioi = np.array([b - a for a, b, p, q in zip(ons[:-1], ons[1:], ph[:-1], ph[1:]) if p == q and b - a > 0.08])
    if ioi.size < 3:
        return 0.6
    P = np.arange(lo, hi, 0.005)
    q = ioi[None, :, None] / P[:, None, None]
    mult = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8])[None, None, :]
    d = np.min(np.abs(q - mult), axis=2)
    score = np.exp(-0.5 * (d / tol) ** 2).sum(axis=1)
    peak = np.r_[False, (score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:]), False]
    good = np.nonzero(peak & (score >= 0.70 * score.max()))[0]
    if good.size == 0:
        good = [int(np.argmax(score))]
    return float(P[good[-1]])


# ---------------------------------------------------------------- the cache
class Reference:
    """Everything the live engine needs about the piece."""
    def __init__(self, F, notes, plan, key, join_r, breaths, bias_live, beat_s, profile_grid=None,
                 profile_u=None, meta=None, gate=None):
        self.F, self.notes, self.plan, self.key = F, notes, plan, key
        self.join_r, self.breaths, self.bias_live, self.beat_s = join_r, breaths, bias_live, beat_s
        self.profile_grid, self.profile_u = profile_grid, profile_u
        self.meta = meta or {}
        self.gate = gate          # None, or the input gate the reference was analysed with (gate_from_reference)

    @property
    def duration(self):
        return float(self.F['t'][-1]) if len(self.F['t']) else 0.0

    def profile(self):
        if self.profile_grid is None:
            return None
        g, u = self.profile_grid, self.profile_u
        return lambda r: float(np.interp(r, g, u))

    def save(self, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v) for k, v in self.F.items()}
        if self.profile_grid is not None:
            arrays['profile_grid'] = self.profile_grid
            arrays['profile_u'] = self.profile_u
        np.savez_compressed(str(prefix) + '.npz', **arrays)
        js = {'version': CACHE_VERSION, 'key': self.key, 'join_r': self.join_r, 'breaths': self.breaths,
              'bias_live': self.bias_live, 'beat_s': self.beat_s, 'notes': self.notes, 'plan': self.plan,
              'has_profile': self.profile_grid is not None, 'meta': self.meta}
        if self.gate is not None:           # absent (= no gate) in every cache made without --gate-db
            js['gate'] = self.gate
        Path(str(prefix) + '.json').write_text(json.dumps(js, indent=1))
        return Path(str(prefix) + '.npz'), Path(str(prefix) + '.json')

    @classmethod
    def load(cls, prefix):
        prefix = str(prefix)
        for ext in ('.npz', '.json'):
            if prefix.endswith(ext):
                prefix = prefix[:-len(ext)]
        js = json.loads(Path(prefix + '.json').read_text())
        if js.get('version') != CACHE_VERSION:
            raise ValueError(f'{prefix}.json: cache version {js.get("version")} != {CACHE_VERSION}; re-run rehearse')
        z = np.load(prefix + '.npz')
        F = {k: z[k] for k in ('t', 'midi', 'voiced', 'db', 'nov')}
        pg = z['profile_grid'] if 'profile_grid' in z.files else None
        pu = z['profile_u'] if 'profile_u' in z.files else None
        plan = [dict(p) for p in js['plan']]
        return cls(F, js['notes'], plan, js['key'], js['join_r'], js['breaths'], js['bias_live'], js['beat_s'],
                   pg, pu, js.get('meta'), gate=js.get('gate'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_notes_file(path, gap=0.05):
    """Reference notes from an annotation file: rows 'onset offset midi' (s, s, MIDI; comma, tab or
    space separated; a header line is skipped). Each note gets the onset kind the Accompanist uses:
    'attack' after a silence of > gap s, 'repeat' for the same pitch again, else 'glide' (legato)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(',', ' ').replace('\t', ' ').split()
        try:
            on, off, m = float(parts[0]), float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            continue
        rows.append((on, off, m))
    rows.sort()
    out = []
    for i, (on, off, m) in enumerate(rows):
        if i == 0 or on - rows[i - 1][1] > gap:
            kind = 'attack'
        elif int(round(m)) == int(round(rows[i - 1][2])):
            kind = 'repeat'
        else:
            kind = 'glide'
        out.append({'on': on, 'off': off, 'midi': int(round(m)), 'kind': kind})
    return out


def rehearse(ref_path, prefix, take=None, plan_json=None, join_at=None, beat=None, log=print, notes_file=None,
             breath_gap=0.25, gate_db=None, bias=None, bias_source=None):
    """Build and cache the reference analysis. Returns the Reference.

    notes_file: use annotated notes of the reference recording (onset, offset, midi) instead of
    segmenting it with offline_notes. On real singing offline_notes over-segments badly (vibrato,
    consonants: 236 detected vs 54 annotated notes on an ECD soprano take), and the Accompanist
    then waits for onsets that never come; with a score or a hand-checked transcription of the
    rehearsal the reference notes are the notes she actually sings.

    gate_db: the evaluation's input level gate (eval/duet_eval/features.py), default None = off.
    The threshold is fixed here from the reference (90th-percentile frame level - gate_db), the
    reference and the --take are analysed after the gate, and the threshold is stored in the cache
    so `run` gates the live input with it (as the evaluation gates both takes).
    bias: {kind: s} detector bias that replaces the synthetic-singer calibration (live_bias), e.g.
    parse_bias('dcs'); also used for the --take profile. Default None = synthetic calibration."""
    t0 = time.perf_counter()
    cal = calibrate_detectors()
    x = load_audio(ref_path)
    gate = gate_from_reference(x, gate_db) if gate_db is not None else None
    F = analyse(gate_audio(x, gate))
    bias_live = dict(bias) if bias is not None else cal['live_bias']
    if notes_file:
        det = read_notes_file(notes_file)            # annotated onsets: no detector delay to remove
    else:
        det = offline_notes(F)
        for nt in det:                               # remove the measured detector delay
            nt['on'] -= cal['ref_bias'].get(nt['kind'], 0.0)
    if plan_json:                                    # e.g. from the harmony model (harmony/)
        pj = json.loads(Path(plan_json).read_text())
        plan, key = pj['plan'], pj.get('key', '?')
    else:
        plan, key = plan_harmony(det)
    if notes_file:          # breaths = real rests between annotated notes (unvoiced consonants are not breaths)
        gaps = [(a['off'], b['on']) for a, b in zip(det[:-1], det[1:]) if b['on'] - a['off'] >= breath_gap]
    else:
        gaps = find_gaps(F)
    breaths = [(a + b) / 2 for a, b in gaps]
    dur = len(x) / SR
    if join_at is None:                              # same rule as run_v0: first attack after 25 %
        join_r = next((nt['on'] for nt in det if nt['on'] >= 0.25 * dur and nt['kind'] == 'attack'),
                      next(nt['on'] for nt in det if nt['on'] >= 0.25 * dur))
    else:
        jr = join_at * dur if join_at <= 1.0 else join_at
        join_r = next(nt['on'] for nt in det if nt['on'] >= jr - 1e-6)
    beat_s = float(beat) if beat else estimate_beat(det, breaths)
    pg = pu = None
    if take:                                         # learn her timing habits from a rehearsal take
        ra = Accompanist(det, [], 0.0, bias_live=bias_live, breaths=breaths)
        _follow(gate_audio(load_audio(take), gate), F, ra)
        prof = profile_from_anchors(ra.anchors, dur)
        pg = np.arange(0.0, dur + 0.01, 0.01)
        pu = np.array([prof(g) for g in pg])
    meta = {'plan_source': 'json' if plan_json else 'rule',
            'notes_source': str(Path(notes_file).resolve()) if notes_file else 'offline_notes',
            'reference': str(Path(ref_path).resolve()), 'reference_sha256': sha256(ref_path),
            'take': str(Path(take).resolve()) if take else None, 'duration_s': dur, 'n_notes': len(det),
            'n_plan': len(plan), 'calibration': cal, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'build_s': None}
    if gate is not None or bias is not None:        # absent in caches made with the defaults
        meta['front_end'] = {'input_gate': gate,
                             'bias_live': None if bias is None else {'used': bias_live, 'source': bias_source,
                                                                      'replaced_synthetic': cal['live_bias']}}
    ref = Reference(F, det, plan, key, float(join_r), breaths, bias_live, beat_s, pg, pu, meta, gate=gate)
    meta['build_s'] = round(time.perf_counter() - t0, 2)
    paths = ref.save(prefix)
    gate_txt = 'off' if gate is None else f"{gate['gate_db']:g} dB (thr {gate['thr_db']:.1f} dBFS)"
    log(f'rehearsed {ref_path}: {dur:.1f} s, {len(det)} notes, key {key}, {len(breaths)} breaths, '
        f'joins at {join_r:.2f} s, beat {beat_s:.3f} s, profile {"yes" if take else "no"}, '
        f'input gate {gate_txt}, '
        f'bias {"synthetic" if bias is None else "override"} '
        f'-> {paths[0]} + {paths[1].name} ({meta["build_s"]} s)')
    return ref


# ---------------------------------------------------------------- harmony plan at run time
_LEARNED = {}


def learned_planner():
    """harmony/model.py plan_harmony_learned (a CRF trained on Bach chorales; numpy only).
    Returns (function, None) or (None, reason it could not be loaded)."""
    if 'fn' in _LEARNED or 'err' in _LEARNED:
        return _LEARNED.get('fn'), _LEARNED.get('err')
    try:
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location('duet_harmony_model', HARMONY_MODEL)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules['duet_harmony_model'] = mod
        spec.loader.exec_module(mod)
        fn = mod.plan_harmony_learned
        fn([{'on': 0.0, 'off': 0.5, 'midi': 69, 'kind': 'attack'}, {'on': 0.5, 'off': 1.0, 'midi': 67, 'kind': 'attack'}])
        _LEARNED['fn'] = fn
    except Exception as ex:                       # missing params file, numpy mismatch, ...
        _LEARNED['err'] = f'{type(ex).__name__}: {ex}'
    return _LEARNED.get('fn'), _LEARNED.get('err')


def choose_plan(ref, mode=None, log=print):
    """Set ref.plan / ref.key for this run.

    mode 'learned': harmony/model.plan_harmony_learned on the rehearsed notes (falls back to
                    'rule' with a warning if the model cannot be loaded);
    mode 'rule':    follow.plan_harmony (diatonic third/sixth below), the phase-1 behaviour;
    mode 'cache':   the plan stored by `rehearse` (e.g. a --plan JSON);
    None (default): 'cache' if the rehearsal was given a --plan JSON, else 'learned'.
    Both planners keep exactly the same melody notes (note, r, dur), only 'h' differs."""
    want = mode or ('cache' if ref.meta.get('plan_source') == 'json' else 'learned')
    used, err = want, None
    t0 = time.perf_counter()
    if want == 'learned':
        fn, err = learned_planner()
        if fn is None:
            log(f'learned harmony unavailable ({err}); using the rule-based plan', flush=True)
            used = 'rule'
        else:
            ref.plan, ref.key = fn(ref.notes)
    if used == 'rule':
        ref.plan, ref.key = plan_harmony(ref.notes)
    ref.plan = [dict(p) for p in ref.plan]
    ref.meta['harmony'] = {'requested': mode, 'used': used, 'key': ref.key, 'n_plan': len(ref.plan),
                           'fallback_reason': err if used != want else None,
                           'plan_ms': round((time.perf_counter() - t0) * 1000, 2)}
    return used
