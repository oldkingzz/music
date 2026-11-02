"""VER3_causal_probe_regtrkst_v1 (verifier, run v5rt_1101, round 3): is rt:r3_X_regtrkst_v1 causal under bench_rt?
Adapts runs_mini/v5rt_1101/verify/causal_probe.py (imported as a module; that file is not edited). Val only (tuned on
val; this bundle has no test split). Writes ONE new JSON (mode 'x') under runs_mini/v5rt_1101/verify/.

--phase std    causal_probe.main() unchanged with SPEC = rt:r3_X_regtrkst_v1 and CHAIN = the winner's code chain:
               24 val songs (6 per dataset, the same fixed sample), T1 cut tests (L 0.1), T2 latency cuts (L 0.05 / 0.1
               / 0.15), T3 plan(q) independence from notes not yet heard (L 0.05 / 0.1 / 0.15), T4 run history (object
               A again after everything, fresh object B in reverse order), T5 tripwires (file / data / os.environ reads
               during simulate), T6 power toys, and the static scan of the chain.
--phase shift  the register adapter (rt_systems.r3_D_lib_v3.RegAdapter) only shifts when her first heard notes lie below
               MIDI 62 (or above 88), which almost never happens for the val songs at their own pitch, so its octave
               shift and replay path (_decide / _feed) are barely exercised by --phase std. Here the same 24 songs are
               moved down 12 and 24 semitones (a low voice; chords and tonic unchanged) and T1 / T2 / T3 / T4 / T5 are
               re-run at L = 0.1 (T3: up to 10 plan points where a note starts in (q - L, q + 0.01) plus 3 others per
               song). Also records the adapter's shift decisions per song at shifts 0, -12 and -24.

  .venv/bin/python runs_mini/v5rt_1101/verify/VER3_causal_probe_regtrkst_v1.py --phase std|shift \
      --out runs_mini/v5rt_1101/verify/<new VER3_*.json>
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v5rt_1101/verify')
import causal_probe as cp  # noqa: E402  (puts the v5_rt / v3 / duet_beat paths in place; bench_rt as cp.B)
import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

ABS = cp.ABS
OUT_DIR = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'verify')
SPEC = 'rt:r3_X_regtrkst_v1'
CHAIN = ['futures/v5_rt/rt_systems/r3_X_regtrkst_v1.py', 'futures/v5_rt/rt_systems/r3_X_lib_v3.py',
         'futures/v5_rt/rt_systems/r3_X_lib_v2.py', 'futures/v5_rt/rt_systems/r3_X_lib_v1.py',
         'futures/v5_rt/rt_systems/r3_D_lib_v3.py', 'futures/v5_rt/rt_systems/r3_E_lib_v2.py',
         'futures/v5_rt/rt_systems/r3_F_lib_v4.py', 'futures/v5_rt/rt_systems/r3_F_lib_v3.py',
         'futures/v5_rt/rt_systems/r3_F_lib_v2.py', 'futures/v5_rt/rt_systems/r3_F_lib_v1.py',
         'futures/v5_rt/rt_systems/r2_lib_v3.py', 'futures/v5_rt/rt_systems/r2_lib_v2.py',
         'futures/v5_rt/rt_systems/r2_lib_v1.py', 'futures/v5_rt/rt_systems/r1_A_core_v3.py',
         'futures/v5_rt/rt_systems/r1_A_core_v2.py', 'futures/v3_nolatency/systems/r2_mix_v2.py',
         'futures/v3_nolatency/systems/m1_filter_v5lo.py', 'futures/v3_nolatency/systems/m3_seq_v4.py',
         'futures/v3_nolatency/systems/m3_seq_v1.py', 'duet_beat/theory.py']
SHIFTS = (-12, -24)
L = 0.1


def pitch_range_wide(song):
    """causal_probe.pitch_range with a lower floor (24) so random futures exist for songs moved down 24 semitones."""
    ms = [m for _, _, m in song['sop']]
    lo, hi = max(24, min(ms) - 7), min(95, max(ms) + 7)
    return lo, max(hi, lo + 7)


def shifted(song, d):
    s2 = copy.deepcopy(song)
    s2['sop'] = [[on, off, int(m) + d] for on, off, m in song['sop']]
    return s2


def reg_info(obj):
    songs = getattr(obj, 'songs', None)
    if not songs:
        return None
    i = songs[-1]
    return {'s': [list(x) for x in i['s']], 'resync': i['resync'], 'first': i['first']}


def phase_shift(out_path, nq=10, nq_o=3):
    t_start = time.time()
    B = cp.B
    res = {'spec': SPEC, 'phase': 'shift', 'shifts': list(SHIFTS), 'L': L, 'seed': cp.SEED, 'split': 'val',
           'tests': {}, 'fail_examples': [], 'reg': {}}
    B.bench3.B2.heard = B.HEARD1
    train = B.bench3.train_sets()
    VAL = B.bench3.eval_sets('val')
    assert list(VAL) == list(cp.DS), list(VAL)
    cp.pitch_range = pitch_range_wide
    TRIP = cp.TRIP
    TRIP.install()
    try:
        TRIP.phase = 'build'
        objA = B.load(SPEC, train)
        TRIP.phase = None
        rng = random.Random(cp.SEED)
        pick = {ds: sorted(rng.sample(range(len(VAL[ds])), 6)) for ds in cp.DS}
        base_songs = [(ds, j, VAL[ds][j]) for ds in cp.DS for j in pick[ds]]
        recA = cp.Rec(objA, L)
        # the adapter's shift at the songs' own pitch (information only; std covers these runs)
        for ds, j, s in base_songs:
            log, err = cp.run(recA, cp.meta_of(s), cp.heard_notes(s), L)
            assert err is None, (ds, j, err)
            res['reg'][f'0|{ds}|{j}'] = reg_info(objA)
        variants = [(d, ds, j, shifted(s, d)) for d in SHIFTS for ds, j, s in base_songs]
        orig = {}
        for d, ds, j, s in variants:
            log, err = cp.run(recA, cp.meta_of(s), cp.heard_notes(s), L)
            assert err is None, (d, ds, j, err)
            orig[(d, ds, j)] = log
            res['reg'][f'{d}|{ds}|{j}'] = reg_info(objA)
        res['orig_decisions'] = sum(len(v) for v in orig.values())
        res['orig_nonnull'] = sum(sum(e[3] is not None for e in v) for v in orig.values())

        def tally(name, ok_inp, ok_dec, extra=None):
            t = res['tests'].setdefault(name, {'cases': 0, 'inp_mismatch': 0, 'dec_mismatch': 0})
            t['cases'] += 1
            t['inp_mismatch'] += not ok_inp
            t['dec_mismatch'] += not ok_dec
            for k, v in (extra or {}).items():
                t[k] = t.get(k, 0) + v

        def fail(**kw):
            if len(res['fail_examples']) < 40:
                res['fail_examples'].append(kw)

        for test in ('T1_cut', 'T2_latency_cut'):
            lat = test != 'T1_cut'
            for d, ds, j, s in variants:
                n0 = cp.heard_notes(s)
                r2 = random.Random(f'{cp.SEED}-VER3-{test}-{d}-{ds}-{j}')
                for ctype, c in cp.cut_times(n0, s['K'], r2):
                    for f in range(2):
                        s2, nk = cp.copy_cut(s, c, L, r2, latency=lat)
                        n2 = cp.heard_notes(s2)
                        if not cp.prefix_ok(n0, n2, c, L, latency=lat):
                            tally(f'{test}_shift{d}', True, True, {'construction_mismatch': 1})
                            continue
                        log2, err = cp.run(recA, cp.meta_of(s2), n2, L)
                        inp, dec, n, nn, first = cp.compare(orig[(d, ds, j)], log2, c, strict=not lat)
                        extra = {'errors_after_cut': int(err is not None), 'decisions': n, 'nonnull': nn}
                        if not lat:
                            extra['dec_mismatch_incl_c'] = int(not cp.compare(orig[(d, ds, j)], log2, c, False)[1])
                        tally(f'{test}_shift{d}', inp, dec, extra)
                        if not (inp and dec):
                            fail(test=test, shift=d, ds=ds, idx=j, cut=c, ctype=ctype, inp_same=inp, dec_same=dec,
                                 first=first)
            print(f'{test} done {time.time() - t_start:.0f}s', flush=True)

        for d, ds, j, s in variants:
            n0 = cp.heard_notes(s)
            r3 = random.Random(f'{cp.SEED}-VER3-T3-{d}-{ds}-{j}')
            first_hp = n0[0][0] + L
            qs = [0.5 * x for x in range(int(2 * s['K']) + 1) if first_hp <= 0.5 * x < s['K']]
            fresh = [q for q in qs if any(q - L < n[0] < q + 0.01 for n in n0)]
            other = [q for q in qs if q not in set(fresh)]
            sel = sorted(r3.sample(fresh, min(nq, len(fresh))) + r3.sample(other, min(nq_o, len(other))))
            for q in sel:
                s2, nchg, ntie = cp.copy_pitch(s, q, L, r3)
                n2 = cp.heard_notes(s2)
                if not cp.prefix_ok(n0, n2, q, L, latency=True):
                    tally(f'T3_plan_q_shift{d}', True, True, {'construction_mismatch': 1})
                    continue
                log2, err = cp.run(recA, cp.meta_of(s2), n2, L, stop=q)
                inp, dec, n, nn, first = cp.compare(orig[(d, ds, j)], log2, q, strict=False)
                po = [e for e in orig[(d, ds, j)] if e[1] == 0 and e[0] == q]
                pc = [e for e in log2 if e[1] == 0 and e[0] == q]
                plan_same = [e[3] for e in po] == [e[3] for e in pc]
                tally(f'T3_plan_q_shift{d}', inp, dec and plan_same,
                      {'plan_q_attack_in_orig': int(bool(po) and po[0][3] is not None), 'pitches_changed': nchg,
                       'errors': int(err is not None)})
                if not (inp and dec and plan_same):
                    fail(test='T3', shift=d, ds=ds, idx=j, q=q, inp_same=inp, dec_same=dec, plan_same=plan_same,
                         first=first)
        print(f'T3 done {time.time() - t_start:.0f}s', flush=True)

        again = {}
        for d, ds, j, s in variants:
            again[(d, ds, j)] = cp.run(recA, cp.meta_of(s), cp.heard_notes(s), L)[0]
        TRIP.phase = 'build'
        objB = B.load(SPEC, train)
        TRIP.phase = None
        recB = cp.Rec(objB, L)
        fresh_logs = {}
        for d, ds, j, s in reversed(variants):
            fresh_logs[(d, ds, j)] = cp.run(recB, cp.meta_of(s), cp.heard_notes(s), L)[0]
        res['tests']['T4_history_shift'] = {
            'songs': len(variants), 'A_again_identical': sum(again[k] == orig[k] for k in orig),
            'B_fresh_reverse_identical': sum(fresh_logs[k] == orig[k] for k in orig)}
        print(f'T4 done {time.time() - t_start:.0f}s', flush=True)
    finally:
        TRIP.uninstall()

    run_hits = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'run'}
    res['tests']['T5_tripwires_shift'] = {'run_hits': run_hits}
    res['build_reads'] = {f'{k[1]} @ {k[2]} (caller {k[3]})': v for k, v in TRIP.hits.items() if k[0] == 'build'}
    res['build_files'] = sorted({os.path.relpath(p, ABS) if p.startswith(ABS) else p for (_, p) in TRIP.paths})
    nz = {d: sum(1 for k, v in res['reg'].items() if k.startswith(f'{d}|') and v and any(x[1] != 0 for x in v['s']))
          for d in (0,) + SHIFTS}
    rs = {d: sum(1 for k, v in res['reg'].items() if k.startswith(f'{d}|') and v and v['resync'] > 0)
          for d in (0,) + SHIFTS}
    res['reg_summary'] = {'songs_with_nonzero_shift': {str(d): n for d, n in nz.items()},
                          'songs_with_replay': {str(d): n for d, n in rs.items()}, 'songs_per_shift': 24}
    ok = all(t.get('dec_mismatch', 0) == 0 and t.get('inp_mismatch', 0) == 0 and t.get('construction_mismatch', 0) == 0
             for name, t in res['tests'].items() if name.startswith(('T1', 'T2', 'T3')))
    ok = ok and all(t.get('dec_mismatch_incl_c', 0) == 0 for name, t in res['tests'].items() if name.startswith('T1'))
    t4 = res['tests']['T4_history_shift']
    ok = ok and t4['A_again_identical'] == t4['songs'] and t4['B_fresh_reverse_identical'] == t4['songs']
    ok = ok and not run_hits
    res['causal_ok_shift'] = bool(ok)
    res['secs'] = round(time.time() - t_start, 1)
    with open(out_path, 'x') as fh:
        json.dump(res, fh, indent=1, default=str)
    for name, t in res['tests'].items():
        print(name, json.dumps(t, default=str)[:400], flush=True)
    print('reg summary:', res['reg_summary'], flush=True)
    print('build files:', res['build_files'], flush=True)
    print('fail examples:', res['fail_examples'][:5], flush=True)
    print(f"causal_ok_shift {res['causal_ok_shift']}  ({res['secs']} s)  -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=('std', 'shift'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    if not out_path.startswith(OUT_DIR + os.sep) or not os.path.basename(out_path).startswith('VER3_') \
            or not out_path.endswith('.json') or os.path.exists(out_path):
        sys.exit(f'--out must be a new VER3_*.json under {OUT_DIR}')
    for rel in CHAIN:
        assert os.path.exists(os.path.join(ABS, rel)), rel
    cp.SPEC = SPEC
    cp.CHAIN = CHAIN
    if a.phase == 'std':
        sys.argv = [cp.__file__, '--out', out_path]
        cp.main()
    else:
        phase_shift(out_path)


if __name__ == '__main__':
    main()
