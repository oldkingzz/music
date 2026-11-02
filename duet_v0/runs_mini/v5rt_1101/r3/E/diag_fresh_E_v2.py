"""diag_fresh_E_v2 (run v5rt_1101, round 3, idea E): where the live fresh clashes, stumbles and harsh time come from.
v2 = v1 with the decision lookup fixed (RTDuet writes plan notes ahead, so its decision list is not in time order:
v2 sorts it before matching) and the song loop's note-start list built once. v1 was never run.
Val only (the systems were tuned on val; this bundle has no test split). Offline: live_bench_rt is imported first, so
its _NoSound tripwire is in sys.modules before anything else; nothing is played.

Runs one system through live_bench_rt's own pieces (make_engine, voice_v1 render, run_engine, score) on the 17 val Bach
chorales for one rendered voice (or the phone_v11 file), with a thin logger around the system that records which branch
of the system made each decision (the system's own stats counter). Every Duet note is tied to the engine decision that
started it (same start time, same note), so each fresh clash / stumble / harsh moment is attributed to a branch:
  plan_attack, hear_post_attack / hear_post_keep (posterior hear), hear_switch / hear_enter / hear_same / hear_kept_*
  (reactive-like hear), hear_reactive (an adapter without stats), reenter (RTDuet re-enters the system's note after her
  rest).
For each hear decision the tracker error is recorded (heard - true pitch of the true note the answered note belongs
to, live_bench_rt.tracker_rows' matching) and whether it is a split fragment (2nd+ answered note of one true note).

  .venv/bin/python runs_mini/v5rt_1101/r3/E/diag_fresh_E_v2.py <spec> --voice sung|sung_low|phone_v11 --out <new json>
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

SR, HOP = LBR.SR, LBR.HOP
OUT_DIR = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/r3/E'
PRI = ('hear_post_attack', 'hear_post_keep', 'hear_post_sil', 'plan_attack', 'plan_silence', 'hear_enter',
       'hear_switch', 'hear_same', 'hear_kept_plan', 'hear_kept_old')


class Logged:
    def __init__(self, s):
        self.s = s
        self.log = []

    def start(self, meta):
        self.log = []
        self.s.start(meta)

    def _snap(self):
        st = getattr(self.s, 'stats', None)
        return dict(st) if st is not None else {}

    def _diff(self, b):
        st = getattr(self.s, 'stats', None)
        if st is None:
            return []
        return sorted(k for k in st if st[k] > b.get(k, 0))

    def plan(self, q, heard):
        b = self._snap()
        r = self.s.plan(q, heard)
        self.log.append({'kind': 'plan', 'q': q, 'r': None if r is None else int(r), 'br': self._diff(b),
                         'last': None if not heard else [heard[-1][0], heard[-1][1], int(heard[-1][2])]})
        return r

    def hear(self, i, onset, midi, prev_dur):
        b = self._snap()
        r = self.s.hear(i, onset, midi, prev_dur)
        self.log.append({'kind': 'hear', 'i': i, 'onset': onset, 'midi': int(midi), 'r': None if r is None else int(r),
                         'br': self._diff(b)})
        return r


def branch(entry, dec):
    if dec['kind'] == 'stop':
        return 'stop'
    if dec['action'] == 'reenter':
        return 'reenter'
    if entry is None:
        return dec['kind'] + '_?'
    br = entry['br']
    if not br:
        return dec['kind'] + ('_reactive' if dec['kind'] == 'hear' else '_none')
    for p in PRI:
        if p in br:
            return p
    return br[0]


def tie(duet, decs):
    """For each Duet note (a, e, h) the index of the engine decision that started it (same start time, same note).
    The decision list is not in time order (plan notes are written ahead), so it is sorted first."""
    order = sorted(range(len(decs)), key=lambda j: decs[j]['t'])
    ts = [decs[j]['t'] for j in order]
    out = []
    for a, e, h in duet:
        p = bisect.bisect_left(ts, a - 0.0015)
        best = None
        while p < len(ts) and ts[p] <= a + 0.0015:
            j = order[p]
            if decs[j]['duet'] == h and decs[j]['kind'] != 'stop':
                best = j
            p += 1
        out.append(best)
    return out


def tracker_err(eng, her, lat):
    """answered note i -> (true note index k or None, heard - true, fragment number within k)."""
    starts = {i: c0 for c0, i in eng.by_start.items()}
    ons = [n[0] for n in her]
    seen = Counter()
    res = {}
    for i, (on_b, _, midi) in enumerate(eng.heard_notes):
        tau = starts[i] + lat['Li_told'] - lat['Li']
        k = bisect.bisect_right(ons, tau + LBR.MATCH_PRE) - 1
        if k >= 0 and tau <= her[k][1] + LBR.MATCH_POST:
            res[i] = (k, int(midi) - int(her[k][2]), seen[k])
            seen[k] += 1
        else:
            res[i] = (None, None, None)
    return res


def run_song(sysw, spec, song, voice, lat):
    T_b = 60.0 / LBR.BPM
    facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
    eng = LBR.make_engine(sysw, spec, facts, LBR.COUNT_IN)
    t0h = eng.t0 + (lat['Lo'] - lat['Lo_told'])
    y, her, vf = LBR.VV.render(song, voice, t0h, T_b)
    x = np.concatenate([np.zeros(int(round(lat['Li'] * SR))), y.astype(np.float64), np.zeros(SR)])
    track, notes, blips, secs = LBR.run_engine(eng, x, lat)
    duet = sorted(notes + blips)
    z = [bt['z'] for bt in song['beats']]
    acc = LBR.new_acc()
    LBR.score(her, duet, z, song['tonic'], t0h, T_b, acc)
    return eng, her, duet, z, acc, t0h, T_b


def analyse_song(sysw, eng, her, duet, z, ton, t0h, T_b, lat, agg, details, sid):
    decs = eng.decisions
    sys_decs = [d for d in decs if d['kind'] != 'stop']
    log = sysw.log
    assert len(sys_decs) == len(log), (len(sys_decs), len(log))
    ent = {id(d): e for d, e in zip(sys_decs, log)}
    trk = tracker_err(eng, her, lat)
    owner = tie(duet, decs)
    src = []
    for j in owner:
        if j is None:
            src.append(('untied', None, None))
            continue
        d = decs[j]
        e = ent.get(id(d))
        info = trk.get(d['i']) if (d['kind'] == 'hear' and d['i'] is not None) else None
        src.append((branch(e, d), info, e))
    b = lambda x: (x - t0h) / T_b  # noqa: E731
    hv = [(on_s, off_s, m, non) for on_s, off_s, m, non, _ in her]
    ons = [n[0] for n in her]
    counted = []
    for on_s, off_s, m, non, noff in her:
        k0 = int(math.floor(non + 1e-9))
        counted.append(non >= LBR.J and k0 < len(z) and z[k0] >= 0)
    start_idx = {}
    for idx, (a, e, h) in enumerate(duet):
        start_idx.setdefault((a, h), idx)
    Ts, H = LBR.step_fn(duet)
    for x, h in LBR.EV.attacks(duet):
        idx = start_idx.get((x, h))
        s, info, e = src[idx] if idx is not None else ('untied', None, None)
        nt = LBR.EV.her_at(x, hv)
        if nt is None:
            agg['att_rest'][s] += 1
            continue
        xb = max(b(x), nt[3])
        kb = int(math.floor(xb))
        if xb < LBR.J or kb >= len(z) or z[kb] < 0:
            continue
        agg['att'][s] += 1
        iv = abs(nt[2] - h) % 12
        fresh = iv in LBR.HARSH
        wrong = not LBR.BR.hit(h, nt[2], ton, z[kb])
        agg['wrong'][s] += wrong
        terr = None if info is None else info[1]
        frag = None if info is None else info[2]
        if info is not None:
            tag = 'split' if frag else ('right' if terr == 0 else 'wrong')
            agg['att_by_tracker'][s + ':' + tag] += 1
        if fresh:
            agg['fresh'][s] += 1
            if info is not None:
                agg['fresh_by_tracker'][s + ':' + tag] += 1
            agg['fresh_above'][s] += h > nt[2]
            details.append({'song': sid, 'beat': round(b(x), 3), 'her_true': nt[2], 'duet': h, 'iv': iv, 'src': s,
                            'terr': terr, 'frag': frag, 'above': bool(h > nt[2]),
                            'call': None if e is None else {k: e[k] for k in e if k != 'br'}})
        if wrong and info is not None:
            agg['wrong_by_tracker'][s + ':' + tag] += 1
        j = bisect.bisect_right(ons, x + LBR.ST_PRE) - 1
        if j >= 0 and counted[j] and x <= ons[j] + LBR.ST_POST:
            q = bisect.bisect_right(Ts, x) - 1
            while q + 1 < len(Ts) and H[q + 1] == h:
                q += 1
            t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
            if t_change - x < LBR.ST_WIN:
                agg['stumble'][s] += 1
    dt = 0.005
    starts = [n[0] for n in duet]
    for (on_s, off_s, m, non, noff), c in zip(her, counted):
        for t in np.arange(on_s, off_s, dt):
            kb = int(math.floor(b(t)))
            if kb < LBR.J or kb >= len(z) or z[kb] < 0:
                continue
            q = bisect.bisect_right(starts, t) - 1
            s = 'silent'
            h = -1
            if q >= 0 and duet[q][0] <= t < duet[q][1]:
                h = duet[q][2]
                s = src[q][0]
            agg['time'][s] += dt
            if h >= 0 and abs(m - h) % 12 in LBR.HARSH:
                agg['harsh_t'][s] += dt
            if LBR.BR.hit(h, m, ton, z[kb]):
                agg['hit_t'][s] += dt
    for d, e in zip(sys_decs, log):
        if d['kind'] != 'hear':
            continue
        bb = branch(e, d)
        info = trk.get(d['i'])
        if info is None or info[0] is None:
            agg['hear_calls'][bb + ':extra'] += 1
            continue
        k, terr, frag = info
        tag = 'split' if frag else ('right' if terr == 0 else 'wrong')
        agg['hear_calls'][bb + ':' + tag] += 1


def phone(sysw, spec, agg, details):
    cfg = LBR.EV.V11
    x = LBR.EV.read(cfg['file']).astype(np.float64)
    facts = {'tonic': cfg['tonic'], 'M': cfg['M'], 'K': cfg['K'], 'bpm': cfg['bpm']}
    eng = LBR.make_engine(sysw, spec, facts, 0)
    track, notes, blips, secs = LBR.run_engine(eng, x, LBR.PHONE_LAT)
    prox = LBR.EV.heard_proxy(eng, track, notes, blips, LBR.EV.LO)
    decs = eng.decisions
    sys_decs = [d for d in decs if d['kind'] != 'stop']
    log = sysw.log
    ent = {id(d): e for d, e in zip(sys_decs, log)}
    duet = sorted(notes + blips)
    owner = tie(duet, decs)
    src = [branch(ent.get(id(decs[j])), decs[j]) if j is not None else 'untied' for j in owner]
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    start_idx = {}
    for idx, (a, e, h) in enumerate(duet):
        start_idx.setdefault((a, h), idx)
    T_b = 60.0 / cfg['bpm']
    for xx, h in LBR.EV.attacks(duet):
        idx = start_idx.get((xx, h))
        s = src[idx] if idx is not None else 'untied'
        nt = LBR.EV.her_at(xx, her)
        if nt is None:
            agg['att_rest'][s] += 1
            continue
        agg['att'][s] += 1
        iv = abs(nt[2] - h) % 12
        if iv in LBR.HARSH:
            agg['fresh'][s] += 1
            details.append({'t': round(xx, 3), 'beat': round((xx - eng.t0) / T_b, 3), 'her_heard': nt[2], 'duet': h,
                            'iv': iv, 'src': s})
    t_out = (np.arange(len(track)) + 2) * HOP / SR + LBR.EV.LO
    starts = [n[0] for n in duet]
    for on, off, m, _ in her:
        for i in np.where((t_out >= on) & (t_out < off))[0]:
            t = t_out[i]
            q = bisect.bisect_right(starts, t) - 1
            s = src[q] if (q >= 0 and duet[q][0] <= t < duet[q][1]) else 'silent'
            agg['time'][s] += 0.01
            if track[i] >= 0 and abs(m - int(track[i])) % 12 in LBR.HARSH:
                agg['harsh_t'][s] += 0.01
    rows = []
    hear_log = [e for e in log if e['kind'] == 'hear']
    for ev, e in zip(eng.events, hear_log):
        rows.append([ev['onset_beats'], ev['her_heard'], ev['duet'], ev['action'], e['br']])
    plan_rows = [[e['q'], e['r'], e['br']] for e in log if e['kind'] == 'plan' and e['r'] is not None]
    her_rows = [[round(on, 3), round(off, 3), m] for on, off, m, _ in her]
    return prox, rows, plan_rows, [[round(a, 3), round(e, 3), int(h), s] for (a, e, h), s in zip(duet, src)], her_rows, \
        eng.t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--voice', required=True, choices=('sung', 'sung_low', 'clean', 'phone_v11'))
    ap.add_argument('--songs', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    if not path.startswith(OUT_DIR + os.sep) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + OUT_DIR)
    sysw = Logged(LBR.load_system(a.spec))
    agg = defaultdict(Counter)
    details = []
    out = {'spec': a.spec, 'voice': a.voice, 'note': 'val only, systems tuned on val; no test split'}
    if a.voice == 'phone_v11':
        prox, rows, plan_rows, duet, her_rows, t0 = phone(sysw, a.spec, agg, details)
        out.update({'proxies': prox, 'hears': rows, 'plans_named': plan_rows, 'duet': duet, 'her_seg': her_rows,
                    't0': t0})
        print('proxies', json.dumps(prox), flush=True)
        for r in rows:
            print('  hear', r, flush=True)
        for r in plan_rows:
            print('  plan', r, flush=True)
        for r in details:
            print('  fresh', r, flush=True)
    else:
        lat = LBR.LAT['wired']
        songs = LBR.bench3.eval_sets('val')['bach']
        if a.songs:
            songs = songs[:a.songs]
        tot = LBR.new_acc()
        for s in songs:
            eng, her, duet, z, acc, t0h, T_b = run_song(sysw, a.spec, s, a.voice, lat)
            for k in LBR.ACC_KEYS:
                tot[k] += acc[k]
            analyse_song(sysw, eng, her, duet, z, s['tonic'], t0h, T_b, lat, agg, details, s['id'])
        out['metrics'] = LBR.metrics(tot)
        m = out['metrics']
        print(f"{a.spec} {a.voice}: rt_nd_live {m['rt_nd_live']:.4f} harsh {m['harsh']:.4f} fresh/100 "
              f"{m['fresh_per100']:.2f} stumbles/100 {m['stumbles_per100']:.2f} notes {m['notes']}", flush=True)
    out['agg'] = {k: dict(v) for k, v in agg.items()}
    out['fresh_details'] = details
    for k in ('att', 'fresh', 'wrong', 'stumble', 'att_rest', 'fresh_above', 'att_by_tracker', 'fresh_by_tracker',
              'wrong_by_tracker', 'hear_calls', 'time', 'harsh_t', 'hit_t'):
        if k in agg:
            v = agg[k]
            print(f"{k:16s} " + ', '.join(f"{kk} {round(vv, 2)}" for kk, vv in sorted(v.items(), key=lambda t: -t[1])),
                  flush=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
