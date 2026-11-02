"""Round 3, idea F (stumble guard) DIAGNOSTIC v2 (v1 + clearer labels), not a scored bench. Val only, tuned on val.
Runs the frozen live bench's own pieces (futures/v5_rt/live_bench_rt.py: make_engine, run_engine, score, metrics;
voice_v1.render) on the val Bach chorales with the decision log kept, and labels every counted attack:
  origin     plan / hear / hear_reenter / none (no decision found at that time)
  near       the attack lies in the stumble window of one of her counted onsets ([on - 60 ms, on + 150 ms])
  pc         the attack against her TRUE note: hit, good (consonant, not a chord tone), dbl (unison/octave), harsh
  above      Duet's note above her true note
  stumble    replaced within 250 ms by: plan / hear_first (the first answer of the same true note: wrong-then-corrected)
             / hear_split (a later answer of the same true note: a tracker split) / hear_next / stop
  path       the stats counters the rt system moved in the call that made the attack (and the one that replaced it)
Offline only: the _NoSound tripwire is installed before any other import; nothing is played.

  .venv/bin/python runs_mini/v5rt_1101/r3/F/diag_live_F_v2.py <spec> [<spec> ...] --voice sung --out <new json>
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
SR = LBR.SR
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
        self.calls.append(('plan', q, r, sorted(self._st() - b)))
        return r

    def hear(self, i, onset, midi, prev_dur):
        b = self._st()
        r = self.s.hear(i, onset, midi, prev_dur)
        self.calls.append(('hear', i, r, sorted(self._st() - b)))
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


def one_song(rec, spec, song, voice, lat, acc_all, rows):
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
        acc_all[k] += acc[k]
    plan_calls = [c for c in rec.calls if c[0] == 'plan']
    hear_calls = [c for c in rec.calls if c[0] == 'hear']
    dec = eng.decisions
    pi = hi = 0
    for d in dec:
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
    first_ans = {}                                       # true note index -> first heard index answering it
    hear_true = {}
    for i in range(len(eng.heard_notes)):
        tau = starts[i] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + LBR.MATCH_PRE) - 1
        hear_true[i] = k
        if k >= 0 and k not in first_ans:
            first_ans[k] = i

    for x_s, h in EV.attacks(duet):
        da = find(x_s, ('enter', 'switch', 'reenter'))
        origin = 'none' if da is None else (da['kind'] + ('_reenter' if da['action'] == 'reenter' else ''))
        nt = EV.her_at(x_s, hv)
        if nt is None:
            continue
        xb = max((x_s - t0h) / T_b, nt[3])
        kb = int(math.floor(xb))
        if xb < BR.J or kb >= len(z) or z[kb] < 0:
            continue
        her_k = ons.index(nt[0])
        heard_err = None
        if da is not None and da['kind'] == 'hear':
            heard_err = int(eng.heard_notes[da['i']][2] - nt[2])
        j = bisect.bisect_right(ons, x_s + LBR.ST_PRE) - 1
        near = bool(j >= 0 and counted[j] and x_s <= ons[j] + LBR.ST_POST)
        st = None
        if near:
            q = bisect.bisect_right(Ts, x_s) - 1
            while q + 1 < len(Ts) and H[q + 1] == h:
                q += 1
            t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
            if t_change - x_s < LBR.ST_WIN:
                dr = find(t_change)
                if dr is None:
                    by = 'none'
                elif dr['kind'] == 'hear':
                    kt = hear_true.get(dr['i'])
                    by = ('hear_first' if first_ans.get(j) == dr['i'] else 'hear_split') if kt == j else 'hear_next'
                else:
                    by = dr['kind']
                new = H[q + 1] if q + 1 < len(H) else -1
                st = {'by': by, 'dt_ms': round(1000 * (t_change - x_s), 1), 'new': int(new),
                      'new_pc': pitch_class(int(new), nt[2], ton, z[kb]), 'by_path': [] if dr is None else dr['path'],
                      'by_heard_err': (int(eng.heard_notes[dr['i']][2] - nt[2]) if dr is not None and dr['kind'] == 'hear'
                                       else None)}
        rows.append({'song': song['id'], 'beat': round((x_s - t0h) / T_b, 3), 'q': None if da is None else da['q'],
                     'her_beat': nt[3], 'her': nt[2], 'h': int(h), 'pc': pitch_class(h, nt[2], ton, z[kb]),
                     'above': bool(h > nt[2]), 'origin': origin, 'near': near,
                     'path': [] if da is None else da['path'], 'heard_err': heard_err, 'stumble': st})


def summarize(rows, n):
    c = {k: Counter() for k in ('attacks', 'fresh', 'wrong', 'st_by', 'st_pc', 'st_path', 'st_by_path', 'fresh_path',
                                'fresh_err', 'plan_ctx', 'above')}
    for r in rows:
        o = r['origin'] + (':near' if r['near'] else ':held')
        c['attacks'][o] += 1
        c['plan_ctx'][o + ':' + r['pc']] += r['origin'] == 'plan'
        c['above'][o + ':' + ('above' if r['above'] else 'below')] += 1
        if r['pc'] == 'harsh':
            c['fresh'][o] += 1
            c['fresh_path'][r['origin'] + ':' + ','.join(r['path'])] += 1
            c['fresh_err'][r['origin'] + ':err' + ('?' if r['heard_err'] is None else str(r['heard_err']))] += 1
        if r['pc'] != 'hit':
            c['wrong'][o] += 1
        st = r['stumble']
        if st is not None:
            c['st_by'][r['origin'] + '->' + st['by']] += 1
            c['st_pc'][r['origin'] + ':' + r['pc'] + '->' + st['by'] + ':' + st['new_pc']] += 1
            c['st_path'][r['origin'] + ':' + ','.join(r['path'])] += 1
            c['st_by_path'][st['by'] + ':' + ','.join(st['by_path'])] += 1
    out = {k: {kk: v for kk, v in sorted(cc.items(), key=lambda kv: -kv[1]) if v} for k, cc in c.items()}
    out['per100'] = {k: {kk: round(100.0 * v / n, 2) for kk, v in out[k].items()} for k in ('attacks', 'fresh', 'wrong')}
    out['stumbles'] = sum(c['st_by'].values())
    return out


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
        acc = LBR.new_acc()
        rows = []
        for s in songs:
            one_song(rec, spec, s, a.voice, lat, acc, rows)
        m = LBR.metrics(acc)
        sm = summarize(rows, m['notes'])
        res['systems'][spec] = {'metrics': m, 'summary': sm, 'rows': rows}
        print(spec, a.voice, 'rt_nd_live %.4f harsh %.4f fresh/100 %.2f stumbles/100 %.2f wrong/100 %.1f notes %d' % (
            m['rt_nd_live'], m['harsh'], m['fresh_per100'], m['stumbles_per100'], m['wrong_att_per100'], m['notes']),
            flush=True)
        for k in ('st_by', 'st_pc', 'st_path', 'st_by_path', 'fresh', 'fresh_path', 'fresh_err', 'plan_ctx', 'above'):
            print('  ' + k, dict(list(sm[k].items())[:14]), flush=True)
        print('  attacks/100', sm['per100']['attacks'], '| wrong/100', sm['per100']['wrong'], flush=True)
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1, default=LBR._js)


if __name__ == '__main__':
    main()
