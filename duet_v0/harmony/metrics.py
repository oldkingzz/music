"""Music-theory metrics for a harmony line sampled at the melody's note onsets.

All metrics work on parallel arrays over the PLAN entries (one harmony note per kept melody
note): s = melody MIDI, h = harmony MIDI, and for chorales the true Bach alto `a` (None where
the alto rests) and the pitch-class set of Bach's four-voice chord at that onset.

Definitions (two-voice view, because Duet sings a duet):
  consonant_2v  : (s - h) mod 12 in {0, 3, 4, 7, 8, 9}  (P4 counts as dissonant, as in
                  two-voice counterpoint)
  consonant_4v  : the same plus the P4 (consonant between upper voices of a full chord)
  parallel 5/8  : consecutive pair where the melody moves and the interval class stays a
                  perfect fifth or a unison/octave (compound intervals included)
  crossing      : h > s
  spacing       : s - h > 12 (more than an octave between soprano and alto)
  range         : h outside the textbook alto range G3-D5 (MIDI 55-74)
  leap          : |h_t - h_{t-1}| > 4 semitones (larger than a major third);
  big leap      : > 7 semitones (larger than a fifth)
  chord tone    : h mod 12 is one of the pitch classes Bach sounds at that onset
"""
import numpy as np

ALTO_RANGE = (55, 74)
CONS_2V = {0, 3, 4, 7, 8, 9}


def line_metrics(s, h, a=None, chord=None):
    """Counts (not rates) so that they can be pooled over many pieces."""
    s = np.asarray(s, int)
    h = np.asarray(h, int)
    T = len(s)
    iv = s - h
    ivc = iv % 12
    c = {'n': T,
         'cons2': int(np.isin(ivc, list(CONS_2V)).sum()),
         'cons4': int(np.isin(ivc, list(CONS_2V | {5})).sum()),
         'cross': int((h > s).sum()),
         'spacing': int((iv > 12).sum()),
         'range': int(((h < ALTO_RANGE[0]) | (h > ALTO_RANGE[1])).sum()),
         'unison': int((iv == 0).sum())}
    if T > 1:
        moving = np.diff(s) != 0
        same = (ivc[1:] == ivc[:-1]) & np.isin(ivc[1:], (0, 7))
        dh = np.abs(np.diff(h))
        c.update({'pairs': T - 1, 'par5': int((moving & same & (ivc[1:] == 7)).sum()),
                  'par8': int((moving & same & (ivc[1:] == 0)).sum()),
                  'leap': int((dh > 4).sum()), 'bigleap': int((dh > 7).sum()),
                  'abs_dh': float(dh.sum()), 'static': int((dh == 0).sum())})
    else:
        c.update({'pairs': 0, 'par5': 0, 'par8': 0, 'leap': 0, 'bigleap': 0, 'abs_dh': 0.0, 'static': 0})
    if a is not None:
        lab = np.array([x is not None for x in a])
        av = np.array([x if x is not None else -999 for x in a])
        c.update({'n_lab': int(lab.sum()), 'exact': int(((h == av) & lab).sum()),
                  'pc': int((((h - av) % 12 == 0) & lab).sum()),
                  'abs_err': float(np.abs(h - av)[lab].sum())})
    if chord is not None:
        c['chordtone'] = int(sum((hh % 12) in set(ch) for hh, ch in zip(h, chord)))
    return c


def pool(counts):
    tot = {}
    for c in counts:
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
    return tot


def rates(tot):
    """Human-readable rates (percent unless noted) from pooled counts."""
    n, pr = max(tot['n'], 1), max(tot['pairs'], 1)
    r = {'consonance_2v_%': 100 * tot['cons2'] / n,
         'consonance_4v_%': 100 * tot['cons4'] / n,
         'parallel_5ths_per_100': 100 * tot['par5'] / pr,
         'parallel_8ves_per_100': 100 * tot['par8'] / pr,
         'crossing_%': 100 * tot['cross'] / n,
         'spacing_gt_octave_%': 100 * tot['spacing'] / n,
         'range_violation_%': 100 * tot['range'] / n,
         'mean_abs_motion_st': tot['abs_dh'] / pr,
         'leaps_gt_M3_%': 100 * tot['leap'] / pr,
         'leaps_gt_P5_%': 100 * tot['bigleap'] / pr,
         'repeated_note_%': 100 * tot['static'] / pr}
    if 'n_lab' in tot:
        nl = max(tot['n_lab'], 1)
        r.update({'alto_exact_%': 100 * tot['exact'] / nl, 'alto_pitchclass_%': 100 * tot['pc'] / nl,
                  'alto_mae_st': tot['abs_err'] / nl})
    if 'chordtone' in tot:
        r['chord_tone_%'] = 100 * tot['chordtone'] / n
    return r


def bootstrap_ci(per_piece, key_num, key_den, n_boot=2000, seed=0, scale=100.0):
    """95 % percentile CI of a pooled ratio, resampling whole pieces."""
    num = np.array([c[key_num] for c in per_piece], float)
    den = np.array([c[key_den] for c in per_piece], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    vals = scale * num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(per_a, per_b, key_num, key_den, n_boot=2000, seed=0, scale=100.0):
    """CI of (A - B) for a pooled ratio, resampling pieces jointly; returns (diff, lo, hi, p_le0)."""
    na = np.array([c[key_num] for c in per_a], float)
    nb = np.array([c[key_num] for c in per_b], float)
    da = np.array([c[key_den] for c in per_a], float)
    db = np.array([c[key_den] for c in per_b], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(na), (n_boot, len(na)))
    d = scale * (na[idx].sum(1) / np.maximum(da[idx].sum(1), 1) - nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1))
    full = scale * (na.sum() / max(da.sum(), 1) - nb.sum() / max(db.sum(), 1))
    return float(full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())
