"""v5rt_1101 phase 1, read-only: what off-grid onsets look like (folk, pop909) and how many of her notes start exactly
on a plan point (a multiple of 0.5 beat), per dataset. Prints only; no titles or artists."""
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v3_nolatency'))
import bench3  # noqa: E402

for ds, songs in bench3.eval_sets('val').items():
    n = on_half = on_int = 0
    sample = []
    for s in songs:
        for on, off, m in sorted(s['sop'], key=lambda x: x[0]):
            n += 1
            on_half += (on / 0.5) == int(on / 0.5)
            on_int += on == int(on)
            if abs(on * 48 - round(on * 48)) > 1e-6 and len(sample) < 12:
                sample.append((repr(on), repr(off)))
    print(f'{ds}: notes {n}, onset on a half beat {on_half} ({on_half / n:.3f}), on an integer beat {on_int} ({on_int / n:.3f}); '
          f'off-grid sample {sample}', flush=True)
