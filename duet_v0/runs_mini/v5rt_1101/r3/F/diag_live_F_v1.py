"""Round 3, idea F (stumble guard) DIAGNOSTIC, not a scored bench: where do live stumbles, fresh clashes and wrong
attacks come from? Runs the frozen live bench's own pieces (futures/v5_rt/live_bench_rt.py: make_engine, run_engine,
score; voice_v1.render) on the val Bach chorales (val only, tuned on val; no test split), keeps the engine's decision
log and labels each event by the decision that made it:
  attack origin  plan (a plan point started it), hear (her answered note), reenter (hear None after a rest)
  replaced by    plan / hear-same-note (a tracker split) / hear-next-note / stop (she stopped)
  pitch          the attack against her TRUE note: hit, good (consonant, not a chord tone), dbl (unison/octave), harsh
  system path    which branch of the rt system answered (stats counters before/after each call)
Offline only: the _NoSound tripwire is installed before any other import; nothing is played.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/diag_live_F_v1.py <spec> [<spec> ...] --voice sung --out <new json>
"""
import sys
import types


class _NoSound(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise RuntimeError('offline diagnostic: no sound device may be used (sounddevice.' + name + ')')


for _mod in ('sounddevice', '_sounddevice'):
    sys.modules[_mod] = _NoSound(_mod)

import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402

import numpy as np  # noqa: E402

BR, EV, VV = LBR.BR, LBR.EV, LBR.VV
GOOD = BR.GOOD
OUT_DIR = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r3/F'


class Rec:
    """Wraps an rt system: records which stats counters each plan / hear call moved."""

    def __init__(self, s):
        self.s = s
        self.calls = []

    def _st(self):
        return Counter(getattr(self.s, 'stats', {}))

    def start(self, meta):
        self.calls = []
        return self.s.start(meta)

    def plan(self, q, heard):
        b = self._st()
        r = self.s.plan(q, heard)
        d = self._st() - b
        self.calls.append(('plan', q, r, sorted(d)))
        return r

    def hear(self, i, onset, midi, prev_dur):
        b = self._st()
        r = self.s.hear(i, onset, midi, prev_dur)
        d = self._st() - b
        self.calls.append(('hear', i, r, sorted(d)))
        return r


def pitch_class(h, m, ton, zk):
    if h < 0:
        return 'sil'
    iv = abs(m - h) % 12
    if iv in BR.HARSH:
        return 'harsh'
    if iv == 0:
        return 'dbl'
    return 'hit' if BR.hit(h, m, ton, zk) else 'good'


def one_song(rec, spec, song, voice, lat, out, rows):
    T_b = 60.0 / LBR.BPM
    facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
    eng = LBR.make_engine(rec, spec, facts, LBR.COUNT_IN)
    t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
    y, her, vf = VV.render(song, voice, t0h, T_b)
    x = np.concatenate([np.zeros(int(round(lat['Li'] * SR))), y.astype(np.float64), np.zeros(SR)])
    track, notes, blips, secs = LBR.run_engine(eng, x, lat)
    duet = sorted(notes + blips)
    z = [bt['z'] for bt in song['beats']]
    ton = song['tonic']
    acc = LBR.new_acc()
    LBR.score(her, duet, z, ton, t0h, T_b, acc)
    for k in LBR.ACC_KEYS:
        out['acc'][k] += acc[k]
    # decisions by time; system calls in order (one per plan / hear decision)
    calls = rec.calls
    plan_calls = [c for c in calls if c[0] == 'plan']
    hear_calls = [c for c in calls if c[0] == 'hear']
    dec = eng.decisions
    pi = hi = 0
    for d in dec:                                       # attach the system path to each decision
        if d['kind'] == 'plan':
            d['path'] = plan_calls[pi][3] if pi < len(plan_calls) else []
            pi += 1
        elif d['kind'] == 'hear':
            d['path'] = hear_calls[hi][3] if hi < len(hear_calls) else []
            hi += 1
        else:
            d['path'] = []
    dts = [d['t'] for d in dec]

    def find(t, acts=None):
        j = bisect.bisect_left(dts, t - 0.0006)
        best = None
        while j < len(dec) and dts[j] <= t + 0.0006:
            if acts is None or dec[j]['action'] in acts:
                best = dec[j]
            j += 1
        return best

    hv = [(on_s, off_s, m, non) for on_s, off_s, m, non, _ in her]
    ons = [n[0] for n in her]
    counted = [non >= BR.J and int(math.floor(non + 1e-9)) < len(z) and z[int(math.floor(non + 1e-9))] >= 0
               for _, _, _, non, _ in her]
    Ts, H = LBR.step_fn(duet)
    starts = {i: c0 for c0, i in eng.by_start.items()}

    def true_idx_of_hear(i):
        tau = starts[i] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + LBR.MATCH_PRE) - 1
        return k

    for x_s, h in EV.attacks(duet):
        da = find(x_s, ('enter', 'switch', 'reenter'))
        origin = 'none' if da is None else (da['kind'] + ('_reenter' if da['action'] == 'reenter' else ''))
        path = [] if da is None else da['path']
        nt = EV.her_at(x_s, hv)
        if nt is None:
            continue
        xb = max((x_s - t0h) / T_b, nt[3])
        kb = int(math.floor(xb))
        if xb < BR.J or kb >= len(z) or z[kb] < 0:
            continue
        pc = pitch_class(h, nt[2], ton, z[kb])
        her_k = ons.index(nt[0])
        heard_err = None
        if da is not None and da['kind'] == 'hear':
            heard_err = int(eng.heard_notes[da['i']][2] - nt[2])
        # stumble?
        st = None
        j = bisect.bisect_right(ons, x_s + LBR.ST_PRE) - 1
        if j >= 0 and counted[j] and x_s <= ons[j] + LBR.ST_POST:
            q = bisect.bisect_right(Ts, x_s) - 1
            while q + 1 < len(Ts) and H[q + 1] == h:
                q += 1
            t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
            if t_change - x_s < LBR.ST_WIN:
                dr = find(t_change)
                if dr is None:
                    by = 'none'
                elif dr['kind'] == 'hear':
                    by = 'hear_split' if true_idx_of_hear(dr['i']) == j else 'hear_next'
                else:
                    by = dr['kind']
                new = H[q + 1] if q + 1 < len(H) else -1
                st = {'by': by, 'dt_ms': round(1000 * (t_change - x_s), 1), 'new': int(new),
                      'new_pc': pitch_class(int(new), nt[2], ton, z[kb]),
                      'by_path': [] if dr is None else dr['path']}
        rows.append({'song': song['id'], 'beat': round((x_s - t0h) / T_b, 3), 'her_beat': nt[3], 'her': nt[2], 'h': int(h),
                     'pc': pc, 'origin': origin, 'path': path, 'heard_err': heard_err, 'counted': bool(counted[her_k]),
                     'stumble': st})


def summarize(rows, n_notes):
    s = {'attacks': Counter(), 'fresh': Counter(), 'wrong': Counter(), 'stumble_origin_by': Counter(),
         'stumble_pc': Counter(), 'stumble_path': Counter(), 'fresh_path': Counter(), 'stumble_by_path': Counter()}
    for r in rows:
        o = r['origin']
        s['attacks'][o] += 1
        if r['pc'] == 'harsh':
            s['fresh'][o] += 1
            s['fresh_path'][o + ':' + ','.join(r['path'])] += 1
        if r['pc'] != 'hit':
            s['wrong'][o] += 1
        if r['stumble'] is not None:
            st = r['stumble']
            s['stumble_origin_by'][o + '->' + st['by']] += 1
            s['stumble_pc'][o + ':' + r['pc'] + '->' + st['new_pc']] += 1
            s['stumble_path'][o + ':' + ','.join(r['path'])] += 1
            s['stumble_by_path'][st['by'] + ':' + ','.join(st['by_path'])] += 1
    out = {k: {kk: v for kk, v in sorted(c.items(), key=lambda kv: -kv[1])} for k, c in s.items()}
    out['per100'] = {k: {kk: round(100.0 * v / n_notes, 2) for kk, v in c.items()}
                     for k, c in (('attacks', s['attacks']), ('fresh', s['fresh']), ('wrong', s['wrong']))}
    out['stumbles_per100'] = round(100.0 * sum(s['stumble_origin_by'].values()) / n_notes, 2)
    return out


SR = LBR.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+')
    ap.add_argument('--voice', required=True, choices=VV.VOICES)
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_DIR + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new .json under {OUT_DIR}/')
    lat = LBR.LAT['wired']
    songs = LBR.bench3.eval_sets('val')['bach']
    if a.songs:
        songs = songs[:a.songs]
    res = {'voice': a.voice, 'lat': 'wired', 'songs': len(songs), 'note': 'val only, tuned on val; diagnostic',
           'systems': {}}
    for spec in a.specs:
        rec = Rec(LBR.load_system(spec))
        out = {'acc': LBR.new_acc()}
        rows = []
        for s in songs:
            one_song(rec, spec, s, a.voice, lat, out, rows)
        m = LBR.metrics(out['acc'])
        summ = summarize(rows, m['notes'])
        res['systems'][spec] = {'metrics': m, 'summary': summ, 'rows': rows}
        print(spec, a.voice, 'rt_nd_live %.4f harsh %.4f fresh/100 %.2f stumbles/100 %.2f wrong/100 %.1f' % (
            m['rt_nd_live'], m['harsh'], m['fresh_per100'], m['stumbles_per100'], m['wrong_att_per100']), flush=True)
        print('  stumbles origin->by', dict(summ['stumble_origin_by']), flush=True)
        print('  stumbles pitch', dict(summ['stumble_pc']), flush=True)
        print('  stumble paths', dict(list(summ['stumble_path'].items())[:8]), flush=True)
        print('  replaced-by paths', dict(list(summ['stumble_by_path'].items())[:8]), flush=True)
        print('  fresh by origin', dict(summ['fresh']), 'paths', dict(list(summ['fresh_path'].items())[:8]), flush=True)
        print('  attacks/100', summ['per100']['attacks'], 'wrong/100', summ['per100']['wrong'], flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LBR._js)


if __name__ == '__main__':
    main()
