"""VER4_causal_probe_hgeh_v1 (verifier, run v5rt_1101, round 4): is rt:r4_X_hgeh_v1 (the round-4 rule pick) causal
under bench_rt at L = 0.1 and L = 0.5? Adapts runs_mini/v5rt_1101/verify/VER3_causal_probe_regtrkst_v1.py; imports
runs_mini/v5rt_1101/verify/causal_probe.py as a module (that file is not edited) and uses bench_rt.simulate unchanged.
Val only (the systems are tuned on val; this bundle has no test split). Writes ONE new JSON (mode 'x') under
runs_mini/v5rt_1101/verify/ (name must start with VER4_).

Why L = 0.5: every round-4 code path (idea H's lag mode and lag-aware plan, late_keep; idea G's large regime, lag-aware
hedge, E hedge, late override) is switched on only when the call order shows a large lag; at L <= 0.3 the pick is W.
So the probe is run at L = 0.1 (W's regime) and L = 0.5 (both lag regimes on), each on the 24 fixed val songs (6 per
dataset, causal_probe's sample) at her own pitch (shift 0) and moved down 12 semitones (shift -12: the register
adapter's octave shift and replay path, which re-starts the core inside a song and re-hears old notes while the lag
state must survive).
  T1 cut      keep true notes with onset < c, random future (notes, durations, chords); PASS = every decision at a time
              < c identical (also reported for <= c) and the inputs bench_rt gave before c identical.
  T2 latency  everything not yet HEARD at c replaced (onset + L > c, and unseen offsets); PASS = decisions at <= c same.
  T3 plan(q)  the true pitch of every note with onset + L > q changed (timing kept); the run stops after q; PASS =
              plan(q) and every earlier decision identical.
  T4 history  every original log again from object A after all T1-T3 runs, and from a fresh object B fed the songs in
              reverse order (shift and L order reversed too); PASS = identical.
  T5 tripwire during simulate: open / io.open / os.open / np.load / torch.load / json.load(s), the bench data loaders,
              os.environ reads (causal_probe.Trip) PLUS time.time / time.time_ns / time.perf_counter /
              time.perf_counter_ns / time.monotonic / time.process_time; a hit counts only when its innermost project
              frame is in the system's code (futures/v5_rt/rt_systems/ or futures/v3_nolatency/systems/); bench_rt's
              own timing calls are listed apart.
  T6 power    causal_probe's toys (a song-memorising cheat must fail T4 / T1, an os.environ / bench-data peek must trip
              T5).
  regime      per original run: whether H's lag mode switched on (LagWrap.info lag_on_at), G's large regime (median of
              the song's q_last - onset samples), and the round-4 counters (lag_plan, lag_attack, late_keep, g_large,
              g_ehedge, g_late_pred, g_late_keep*, g_late_attack, doubt_noatt): the L = 0.5 runs must exercise the new
              code or the probe proves nothing about it.
  static      causal_probe.static_scan of the whole chain plus a scan for time / frame / inspect / gc / latency names.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER4_causal_probe_hgeh_v1.py \
      --out runs_mini/v5rt_1101/verify/VER4_causal_probe_hgeh_v1.json
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/verify')
import causal_probe as cp  # noqa: E402  (puts the v5_rt / v3 / duet_beat paths in place; bench_rt as cp.B)
import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

ABS = cp.ABS
OUT_DIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify')
SPEC = 'rt:r4_X_hgeh_v1'
CHAIN = ['futures/v5_rt/rt_systems/r4_X_hgeh_v1.py', 'futures/v5_rt/rt_systems/r4_X_lib_v1.py',
         'futures/v5_rt/rt_systems/r4_H_lib_v2.py', 'futures/v5_rt/rt_systems/r4_H_lib_v1.py',
         'futures/v5_rt/rt_systems/r4_G_lib_v4.py', 'futures/v5_rt/rt_systems/r4_G_lib_v3.py',
         'futures/v5_rt/rt_systems/r4_G_lib_v2.py', 'futures/v5_rt/rt_systems/r4_G_lib_v1.py',
         'futures/v5_rt/rt_systems/r3_X_lib_v3.py', 'futures/v5_rt/rt_systems/r3_X_lib_v2.py',
         'futures/v5_rt/rt_systems/r3_X_lib_v1.py', 'futures/v5_rt/rt_systems/r3_D_lib_v3.py',
         'futures/v5_rt/rt_systems/r3_E_lib_v2.py', 'futures/v5_rt/rt_systems/r3_F_lib_v4.py',
         'futures/v5_rt/rt_systems/r3_F_lib_v3.py', 'futures/v5_rt/rt_systems/r3_F_lib_v2.py',
         'futures/v5_rt/rt_systems/r3_F_lib_v1.py', 'futures/v5_rt/rt_systems/r2_lib_v3.py',
         'futures/v5_rt/rt_systems/r2_lib_v2.py', 'futures/v5_rt/rt_systems/r2_lib_v1.py',
         'futures/v5_rt/rt_systems/r1_A_core_v3.py', 'futures/v5_rt/rt_systems/r1_A_core_v2.py',
         'futures/v3_nolatency/systems/r2_mix_v2.py', 'futures/v3_nolatency/systems/m1_filter_v5lo.py',
         'futures/v3_nolatency/systems/m3_seq_v4.py', 'futures/v3_nolatency/systems/m3_seq_v1.py',
         'duet_beat/theory.py']
LS = (0.1, 0.5)
SHIFTS = (0, -12)
NQ = {0: (30, 8), -12: (10, 3)}
SYS_DIRS = ('futures/v5_rt/rt_systems/', 'futures/v3_nolatency/systems/')
KEYS = ('lag_plan', 'lag_attack', 'lag_sil', 'late_keep', 'late_switch', 'g_large', 'g_ehedge', 'g_late_pred',
        'g_late_keep', 'g_late_keep_harsh', 'g_late_sil', 'g_late_attack', 'doubt_noatt', 'trk_keep', 'trk_sil')
EXTRA_NAMES = ('time', 'perf_counter', 'monotonic', 'process_time', 'time_ns', 'datetime', '_getframe', 'f_back',
               'f_locals', 'f_globals', 'inspect', 'gc', 'get_referrers', 'latency', 'Li', 'Lo', 'Li_told',
               'Lo_told', 'environ', 'getenv', 'sounddevice', 'socket', 'subprocess', 'urllib', 'requests')


def pitch_range_wide(song):
    """causal_probe.pitch_range with a lower floor (24) so random futures exist for songs moved down."""
    ms = [m for _, _, m in song['sop']]
    lo, hi = max(24, min(ms) - 7), min(95, max(ms) + 7)
    return lo, max(hi, lo + 7)


def shifted(song, d):
    if d == 0:
        return song
    s2 = copy.deepcopy(song)
    s2['sop'] = [[on, off, int(m) + d] for on, off, m in song['sop']]
    return s2


def extra_static():
    out = {}
    for rel in CHAIN:
        with open(os.path.join(ABS, rel)) as fh:
            tree = ast.parse(fh.read())
        sus = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.alias):
                name = node.name.split('.')[0]
            else:
                continue
            if name in EXTRA_NAMES:
                sus.setdefault(name, []).append(getattr(node, 'lineno', None))
        out[rel] = sus
    return out


def install_time_trips():
    saved = []
    for attr in ('time', 'time_ns', 'perf_counter', 'perf_counter_ns', 'monotonic', 'process_time'):
        fn = getattr(time, attr)
        saved.append((time, attr, fn))
        setattr(time, attr, cp.TRIP._wrap(f'time.{attr}', fn))
    return saved


def regime(obj, before):
    core = obj.core
    st = core.stats
    lob = list(getattr(core, 'g_lob', []))
    med = statistics.median(lob[-16:]) if len(lob) >= 3 else None
    return {'h_lag_on_at': obj.info.get('lag_on_at'), 'h_n_d': obj.info.get('n_d'),
            'g_samples': len(lob), 'g_median_last16': med,
            'counts': {k: st.get(k, 0) - before.get(k, 0) for k in KEYS if st.get(k, 0) - before.get(k, 0)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    if not out_path.startswith(OUT_DIR + os.sep) or not os.path.basename(out_path).startswith('VER4_') \
            or not out_path.endswith('.json') or os.path.exists(out_path):
        sys.exit(f'--out must be a new VER4_*.json under {OUT_DIR}')
    for rel in CHAIN:
        assert os.path.exists(os.path.join(ABS, rel)), rel
    cp.SPEC = SPEC
    cp.CHAIN = CHAIN
    cp.pitch_range = pitch_range_wide
    t_start = time.time()
    B = cp.B
    res = {'spec': SPEC, 'Ls': list(LS), 'shifts': list(SHIFTS), 'seed': cp.SEED, 'split': 'val',
           'note': 'val only, systems tuned on val, no test split', 'tests': {}, 'fail_examples': [], 'regime': {}}
    with open(B.__file__, 'rb') as fh:
        import hashlib
        res['bench_sha256'] = hashlib.sha256(fh.read()).hexdigest()
    res['static'] = cp.static_scan()
    res['static_extra'] = extra_static()
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    VAL = B.bench3.eval_sets('val')
    assert list(VAL) == list(cp.DS), list(VAL)
    TRIP = cp.TRIP
    TRIP.install()
    tsaved = install_time_trips()
    try:
        TRIP.phase = 'build'
        objA = B.load(SPEC, train)
        TRIP.phase = None
        rng = random.Random(cp.SEED)
        pick = {ds: sorted(rng.sample(range(len(VAL[ds])), 6)) for ds in cp.DS}
        base_songs = [(ds, j, VAL[ds][j]) for ds in cp.DS for j in pick[ds]]
        res['songs'] = [{'ds': ds, 'idx': j, 'notes': len(s['sop']), 'K': s['K']} for ds, j, s in base_songs]
        variants = [(d, ds, j, shifted(s, d)) for d in SHIFTS for ds, j, s in base_songs]
        orig = {}
        for L in LS:
            recA = cp.Rec(objA, L)
            for d, ds, j, s in variants:
                before = dict(objA.core.stats)
                log, err = cp.run(recA, cp.meta_of(s), cp.heard_notes(s), L)
                assert err is None, (L, d, ds, j, err)
                orig[(L, d, ds, j)] = log
                res['regime'][f'L{L}|{d}|{ds}|{j}'] = regime(objA, before)
            print(f'orig L={L} done {time.time() - t_start:.0f}s', flush=True)
        res['orig_decisions'] = {f'L{L}': sum(len(v) for k, v in orig.items() if k[0] == L) for L in LS}
        res['orig_nonnull'] = {f'L{L}': sum(sum(e[3] is not None for e in v) for k, v in orig.items() if k[0] == L)
                               for L in LS}
        res['orig_plan_attacks'] = {f'L{L}': sum(sum(e[1] == 0 and e[3] is not None for e in v)
                                                 for k, v in orig.items() if k[0] == L) for L in LS}

        def tally(name, ok_inp, ok_dec, extra=None):
            t = res['tests'].setdefault(name, {'cases': 0, 'inp_mismatch': 0, 'dec_mismatch': 0})
            t['cases'] += 1
            t['inp_mismatch'] += not ok_inp
            t['dec_mismatch'] += not ok_dec
            for k, v in (extra or {}).items():
                t[k] = t.get(k, 0) + v

        def fail(**kw):
            if len(res['fail_examples']) < 60:
                res['fail_examples'].append(kw)

        for L in LS:
            recA = cp.Rec(objA, L)
            for test in ('T1_cut', 'T2_latency_cut'):
                lat = test != 'T1_cut'
                for d, ds, j, s in variants:
                    n0 = cp.heard_notes(s)
                    r2 = random.Random(f'{cp.SEED}-VER4-{test}-{L}-{d}-{ds}-{j}')
                    for ctype, c in cp.cut_times(n0, s['K'], r2):
                        for f in range(2):
                            s2, nk = cp.copy_cut(s, c, L, r2, latency=lat)
                            n2 = cp.heard_notes(s2)
                            name = f'{test}_L{L}_shift{d}'
                            if not cp.prefix_ok(n0, n2, c, L, latency=lat):
                                tally(name, True, True, {'construction_mismatch': 1})
                                continue
                            log2, err = cp.run(recA, cp.meta_of(s2), n2, L)
                            inp, dec, n, nn, first = cp.compare(orig[(L, d, ds, j)], log2, c, strict=not lat)
                            extra = {'errors_after_cut': int(err is not None), 'decisions': n, 'nonnull': nn,
                                     f'cuts_{ctype}': 1}
                            if not lat:
                                extra['dec_mismatch_incl_c'] = int(not cp.compare(orig[(L, d, ds, j)], log2, c,
                                                                                  False)[1])
                            else:
                                extra['replaced_started_before_c'] = sum(c - L < x[0] <= c for x in n2[nk:])
                            if err is not None:
                                res.setdefault('errors', []).append({'test': test, 'L': L, 'shift': d, 'ds': ds,
                                                                     'idx': j, 'cut': c, 'err': err[:200]})
                            tally(name, inp, dec, extra)
                            if not (inp and dec):
                                fail(test=test, L=L, shift=d, ds=ds, idx=j, cut=c, ctype=ctype, inp_same=inp,
                                     dec_same=dec, first=first)
                print(f'{test} L={L} done {time.time() - t_start:.0f}s', flush=True)
            for d, ds, j, s in variants:
                nq, nq_o = NQ[d]
                n0 = cp.heard_notes(s)
                r3 = random.Random(f'{cp.SEED}-VER4-T3-{L}-{d}-{ds}-{j}')
                first_hp = n0[0][0] + L
                qs = [0.5 * x for x in range(int(2 * s['K']) + 1) if first_hp <= 0.5 * x < s['K']]
                fresh = [q for q in qs if any(q - L < n[0] < q + 0.01 for n in n0)]
                other = [q for q in qs if q not in set(fresh)]
                sel = sorted(r3.sample(fresh, min(nq, len(fresh))) + r3.sample(other, min(nq_o, len(other))))
                for q in sel:
                    s2, nchg, ntie = cp.copy_pitch(s, q, L, r3)
                    n2 = cp.heard_notes(s2)
                    name = f'T3_plan_q_L{L}_shift{d}'
                    if not cp.prefix_ok(n0, n2, q, L, latency=True):
                        tally(name, True, True, {'construction_mismatch': 1})
                        continue
                    log2, err = cp.run(recA, cp.meta_of(s2), n2, L, stop=q)
                    inp, dec, n, nn, first = cp.compare(orig[(L, d, ds, j)], log2, q, strict=False)
                    po = [e for e in orig[(L, d, ds, j)] if e[1] == 0 and e[0] == q]
                    pc = [e for e in log2 if e[1] == 0 and e[0] == q]
                    plan_same = [e[3] for e in po] == [e[3] for e in pc]
                    tally(name, inp, dec and plan_same,
                          {'plan_q_attack_in_orig': int(bool(po) and po[0][3] is not None),
                           'pitches_changed': nchg, 'ties_onset_plus_L_eq_q': ntie, 'errors': int(err is not None)})
                    if not (inp and dec and plan_same):
                        fail(test='T3', L=L, shift=d, ds=ds, idx=j, q=q, inp_same=inp, dec_same=dec,
                             plan_same=plan_same, first=first)
            print(f'T3 L={L} done {time.time() - t_start:.0f}s', flush=True)

        again = {}
        for L in LS:
            recA = cp.Rec(objA, L)
            for d, ds, j, s in variants:
                again[(L, d, ds, j)] = cp.run(recA, cp.meta_of(s), cp.heard_notes(s), L)[0]
        TRIP.phase = 'build'
        objB = B.load(SPEC, train)
        TRIP.phase = None
        fresh_logs = {}
        for L in reversed(LS):
            recB = cp.Rec(objB, L)
            for d, ds, j, s in reversed(variants):
                fresh_logs[(L, d, ds, j)] = cp.run(recB, cp.meta_of(s), cp.heard_notes(s), L)[0]
        for L in LS:
            for d in SHIFTS:
                ks = [k for k in orig if k[0] == L and k[1] == d]
                res['tests'][f'T4_history_L{L}_shift{d}'] = {
                    'songs': len(ks), 'A_again_identical': sum(again[k] == orig[k] for k in ks),
                    'B_fresh_reverse_identical': sum(fresh_logs[k] == orig[k] for k in ks)}
        print(f'T4 done {time.time() - t_start:.0f}s', flush=True)

        toy_songs = base_songs[:2]
        tm = cp.Rec(cp.ToyMemo(), 0.1)
        first_logs = {(ds, j): cp.run(tm, cp.meta_of(s), cp.heard_notes(s), 0.1)[0] for ds, j, s in toy_songs}
        second = {(ds, j): cp.run(tm, cp.meta_of(s), cp.heard_notes(s), 0.1)[0] for ds, j, s in toy_songs}
        ds, j, s = toy_songs[0]
        n0 = cp.heard_notes(s)
        c = cp.cut_times(n0, s['K'], random.Random(1))[1][1]
        s2, _ = cp.copy_cut(s, c, 0.1, random.Random(2), latency=False)
        cp.ToyMemo.mem = {}
        base = cp.run(tm, cp.meta_of(s), n0, 0.1)[0]
        pert = cp.run(tm, cp.meta_of(s2), cp.heard_notes(s2), 0.1)[0]
        res['tests']['T6_power'] = {
            'toy_memo_T4_detects': sum(first_logs[k] != second[k] for k in first_logs) == len(first_logs),
            'toy_memo_T1_detects': not cp.compare(base, pert, c, strict=True)[1]}
        before = sum(v for k, v in TRIP.hits.items() if k[0] == 'run' and 'causal_probe' in k[2])
        cp.run(cp.Rec(cp.ToyPeek(), 0.1), cp.meta_of(s), n0, 0.1)
        after = sum(v for k, v in TRIP.hits.items() if k[0] == 'run' and 'causal_probe' in k[2])
        res['tests']['T6_power']['toy_peek_trips_T5'] = after > before
    finally:
        for mod, attr, fn in reversed(tsaved):
            setattr(mod, attr, fn)
        TRIP.uninstall()

    run_hits = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'run'}
    sys_hits = {k: v for k, v in run_hits.items() if any(f' @ {p}' in k for p in SYS_DIRS)}
    other = {k: v for k, v in run_hits.items() if k not in sys_hits}
    res['tests']['T5_tripwires'] = {'run_hits_in_system_code': sys_hits, 'run_hits_elsewhere': other}
    res['build_reads'] = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'build'}
    res['build_files'] = sorted({os.path.relpath(p, ABS) if p.startswith(ABS) else p for (_, p) in TRIP.paths})
    reg = res['regime']
    res['regime_summary'] = {}
    for L in LS:
        for d in SHIFTS:
            rows = [v for k, v in reg.items() if k.startswith(f'L{L}|{d}|')]
            tot = {}
            for v in rows:
                for kk, n in v['counts'].items():
                    tot[kk] = tot.get(kk, 0) + n
            res['regime_summary'][f'L{L}_shift{d}'] = {
                'songs': len(rows), 'h_lag_on_songs': sum(v['h_lag_on_at'] is not None for v in rows),
                'g_large_songs': sum(v['g_median_last16'] is not None and v['g_median_last16'] >= 0.5 for v in rows),
                'counts': tot}
    ok = all(t.get('dec_mismatch', 0) == 0 and t.get('inp_mismatch', 0) == 0 and t.get('construction_mismatch', 0) == 0
             for name, t in res['tests'].items() if name.startswith(('T1', 'T2', 'T3')))
    ok = ok and all(t.get('dec_mismatch_incl_c', 0) == 0 for name, t in res['tests'].items() if name.startswith('T1'))
    ok = ok and all(t['A_again_identical'] == t['songs'] and t['B_fresh_reverse_identical'] == t['songs']
                    for name, t in res['tests'].items() if name.startswith('T4'))
    ok = ok and not sys_hits
    p6 = res['tests']['T6_power']
    res['probe_power_ok'] = bool(p6['toy_memo_T4_detects'] and p6['toy_memo_T1_detects'] and p6['toy_peek_trips_T5'])
    l5 = res['regime_summary'].get('L0.5_shift0', {})
    res['new_code_exercised_at_L0.5'] = bool(l5.get('h_lag_on_songs', 0) > 0 and l5.get('g_large_songs', 0) > 0)
    res['causal_ok'] = bool(ok)
    res['secs'] = round(time.time() - t_start, 1)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1, default=str)
    for name, t in res['tests'].items():
        print(name, json.dumps(t, default=str)[:500], flush=True)
    print('regime:', json.dumps(res['regime_summary'])[:2500], flush=True)
    print('build files:', res['build_files'], flush=True)
    print('static extra hits:', {k: v for k, v in res['static_extra'].items() if v}, flush=True)
    print('errors:', len(res.get('errors', [])), (res.get('errors') or [])[:3], flush=True)
    print('fail examples:', res['fail_examples'][:5], flush=True)
    print(f"causal_ok {res['causal_ok']} probe_power_ok {res['probe_power_ok']} new_code_exercised_at_L0.5 "
          f"{res['new_code_exercised_at_L0.5']} ({res['secs']} s) -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
