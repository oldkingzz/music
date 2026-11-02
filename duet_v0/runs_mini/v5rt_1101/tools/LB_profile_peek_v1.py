"""Print a tracker profile JSON's breakdowns compactly (read-only).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_profile_peek_v1.py <profile json>
"""
import json
import sys

p = json.load(open(sys.argv[1]))
print(p['split'], p['label'], 'songs', p['songs'])
for v, prof in list(p['profiles'].items()) + [('pooled', p['pooled'])]:
    print(f'=== {v}')
    for sec in ('overall', 'by_first_vs_rest', 'by_duration_beats', 'by_interval_from_previous', 'by_register_true_midi'):
        groups = {'all': prof[sec]} if sec == 'overall' else prof[sec]
        print(f'  -- {sec}')
        for k, g in groups.items():
            top = sorted(((int(d), s) for d, s in g['heard_minus_true_share'].items() if d != '0'), key=lambda x: -x[1])[:4]
            print(f"    {k:14s} n {g['n']:5d} missed {g['p_missed']:.3f} split {g['p_split']:.3f} "
                  f"right {g['p_pitch_right']:.3f} pc_wrong {g['p_pc_wrong_of_answered'] or 0:.3f} "
                  f"detect {g['detect_ms_median'] or 0:5.1f}/{g['detect_ms_p90'] or 0:5.1f} ms "
                  f"snap_exact {g['snap_err_exact'] or 0:.3f} top errs " + ' '.join(f'{d:+d}:{s:.3f}' for d, s in top))
for v, t in p['tracker_summary'].items():
    print(v, 'onset_err_ms', t['onset_err_ms'], 'snapped', {k: t['snapped_err_beats'][k] for k in ('exact', 'mean_abs')},
          'extra', t['extra_answered'])
