"""why_a: controlled variants of m1_filter_v5lo's Model/Filter (m1's chooser unchanged), real time AND offline
smoothing, val only. (i) transitions, (ii) emission sharpness lam and kappa, (iv) the current-note assumption.
Writes runs_mini/v4_1101/diag/why/why_a_v1.json (mode 'x'). Stops early (saving what it has) after 520 s."""
import sys
import time

import numpy as np

from why_lib import (DS, EXPECT_RT, EXPECT_SMOOTH, FilterV, ModelV, bench3, line, m1, prep, pstay, run, save,
                     smooth, sticky)

NAME = 'why_a_v1'


class Hyb:
    """pooled (m1) transitions + another model's emission."""

    def __init__(self, base, em):
        self.A, self.pi, self.logB, self.Bc = base.A, base.pi, em.logB, em.Bc


def main():
    t0 = time.time()
    train = bench3.train_sets()
    base = ModelV(train)
    ref = m1.build(train).m
    same = bool(np.array_equal(base.A, ref.A) and np.array_equal(base.logB, ref.logB) and np.array_equal(base.pi, ref.pi))
    dsm = {ds: ModelV(train, only={ds}) for ds in DS}
    bal0 = ModelV(train, bal=0)
    kap = {k: ModelV(train, kappa=k) for k in (0.05, 5.0, 50.0)}
    dsA = {ds: dsm[ds].A for ds in DS}
    dsB = {ds: Hyb(base, dsm[ds]) for ds in DS}
    st = {s: sticky(base.A, s) for s in (0.70, 0.85, 0.95)}
    data = prep()
    print('model == m1.Model:', same, f'prep {time.time() - t0:.1f}s', flush=True)

    # name, group, models, tables, lam, cur_w, cur_mode, do_smooth
    V = [('base', 'm1', base, None, 1.5, 1.0, 'eob', True),
         ('T_bal0', 'i', bal0, None, 1.5, 1.0, 'eob', True),
         ('T_dsA_oracle', 'i', base, dsA, 1.5, 1.0, 'eob', True),
         ('T_sticky70', 'i', base, st[0.70], 1.5, 1.0, 'eob', True),
         ('T_sticky85', 'i', base, st[0.85], 1.5, 1.0, 'eob', True),
         ('T_sticky95', 'i', base, st[0.95], 1.5, 1.0, 'eob', True),
         ('E_lam0.5', 'ii', base, None, 0.5, 1.0, 'eob', True),
         ('E_lam1.0', 'ii', base, None, 1.0, 1.0, 'eob', True),
         ('E_lam2.0', 'ii', base, None, 2.0, 1.0, 'eob', True),
         ('E_kappa0.05', 'ii', kap[0.05], None, 1.5, 1.0, 'eob', True),
         ('E_kappa5', 'ii', kap[5.0], None, 1.5, 1.0, 'eob', True),
         ('E_kappa50', 'ii', kap[50.0], None, 1.5, 1.0, 'eob', True),
         ('C_curw0', 'iv', base, None, 1.5, 0.0, 'eob', False),
         ('C_curw0.5', 'iv', base, None, 1.5, 0.5, 'eob', False),
         ('C_curw1.5', 'iv', base, None, 1.5, 1.5, 'eob', False),
         ('C_cur_true_oracle', 'iv', base, None, 1.5, 1.0, 'true', False),
         ('C_cur_median', 'iv', base, None, 1.5, 1.0, 'med', False),
         ('X_dsB_oracle', 'i+', dsB, None, 1.5, 1.0, 'eob', True),
         ('X_dsAB_oracle', 'i+', dsm, None, 1.5, 1.0, 'eob', True),
         ('X_dsA_lam1.0', 'i+ii', base, dsA, 1.0, 1.0, 'eob', True),
         ('X_dsA_lam2.0', 'i+ii', base, dsA, 2.0, 1.0, 'eob', True),
         ('X_sticky85_lam1.0', 'i+ii', base, st[0.85], 1.0, 1.0, 'eob', True),
         ('X_sticky85_lam0.5', 'i+ii', base, st[0.85], 0.5, 1.0, 'eob', True),
         ('X_dsA_curtrue', 'i+iv', base, dsA, 1.5, 1.0, 'true', False)]
    out = {'what': __doc__.strip().split('\n')[0], 'model_equals_m1': same,
           'pstay_tables': {'m1_pooled_bal1': pstay(base.A), 'bal0': pstay(bal0.A),
                            **{f'train_{ds}': pstay(dsm[ds].A) for ds in DS},
                            **{f'sticky{int(s * 100)}': pstay(a) for s, a in st.items()}},
           'tkey_note': 'tkey 0-3 = 4/4 beats 1-4; 4 = downbeat of other meters; 5 = their other beats',
           'variants': {}, 'notes': 'val only (no test split); headline = mean over bach, folk, pop, pop909'}
    print('P(stay) per tkey:', out['pstay_tables'], flush=True)
    for name, grp, mo, tb, lam, cw, cm, dosm in V:
        if time.time() - t0 > 520:
            out['stopped_early_before'] = name
            print('time guard: stopping before', name, flush=True)
            break
        fn = (lambda mo_, lam=lam, cw=cw, cm=cm: FilterV(mo_, lam, cw, cm))
        rt = run(data, mo, fn, tables=tb)
        rec = {'group': grp, 'lam': lam, 'cur_w': cw, 'cur_mode': cm, 'rt': rt}
        print(line(name + ' rt', rt), f'{time.time() - t0:.0f}s', flush=True)
        if dosm:
            sm_fn = (lambda mo_, A_, d, lam=lam: smooth(mo_, mo_.A if A_ is None else A_, lam, d['hrel'], d['pos'],
                                                          d['M'])[0])
            sm = run(data, mo, fn, tables=tb, smooth_fn=sm_fn)
            rec['smooth'] = sm
            print(line(name + ' smooth', sm), f'{time.time() - t0:.0f}s', flush=True)
        out['variants'][name] = rec
    b = out['variants']['base']
    out['reproduces_m1_rt'] = abs(b['rt']['head']['music_nd'] - EXPECT_RT) < 1e-9
    out['reproduces_diag_smooth'] = abs(b['smooth']['head']['music_nd'] - EXPECT_SMOOTH) < 1e-9
    out['secs'] = round(time.time() - t0, 1)
    print('reproduces m1 rt:', out['reproduces_m1_rt'], ' smooth:', out['reproduces_diag_smooth'], out['secs'], 's')
    print('wrote', save(NAME, out))


if __name__ == '__main__':
    sys.exit(main())
