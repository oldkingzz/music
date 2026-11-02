"""Phase D read-only check: build the 6 rD_hist_* wrappers on the train split and confirm the history window
actually reaches both guessers (HMM Filter.H and every GRU sub-model's hist), then feed one synthetic
melody (no eval data) and report how many notes each GRU still holds in its window. Prints only."""
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import importlib  # noqa: E402

import bench3  # noqa: E402

assert not any(k.startswith(('M1_', 'M3_', 'DUET_HIST')) for k in os.environ), 'env knobs set'
train = bench3.train_sets()
K = 64
meta = {'tonic': 0, 'M': 4, 'pos': [k % 4 for k in range(K)], 'K': K}
scale = [0, 2, 4, 5, 7, 9, 11]
for name in ['rD_hist_m1_h0_v1', 'rD_hist_m1_h4_v1', 'rD_hist_m1_h16_v1',
             'rD_hist_mix_h0_v1', 'rD_hist_mix_h4_v1', 'rD_hist_mix_h16_v1']:
    mod = importlib.import_module(f'systems.{name}')
    s = mod.build(train)
    ens = getattr(s, 'ens', None)
    hists = [d.hist for d in ens.subs] if ens is not None else None
    s.start(meta)
    maxbuf, outs = 0, []
    for i in range(100):
        on = 0.5 * i
        outs.append(s.step(i, on, 72 + scale[i % 7], None if i == 0 else 0.5))
        if ens is not None:
            maxbuf = max(maxbuf, max(len(d.buf) for d in ens.subs))
    print(f'{name}: module H={mod.H} filter.H={s.H} gru_hists={hists} max_gru_window_notes={maxbuf if ens is not None else "-"} '
          f'notes_out={sum(o >= 0 for o in outs)}/100')
assert not any(k.startswith(('M1_', 'M3_', 'DUET_HIST')) for k in os.environ), 'env knobs set during check'
print('env knobs still empty: ok')
