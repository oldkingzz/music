"""Print a live_bench_rt JSON's tracker errors: the pitch error histogram and every wrongly heard note with its context
(nominal onset, duration, interval from her previous note, detect lag, raw onset error). Read-only.

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_peek_v1.py <live_bench json> [max rows]
"""
import json
import sys

r = json.load(open(sys.argv[1]))
mx = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print({k: r[k] for k in ('spec', 'voice', 'lat')})
if 'tracker' in r:
    for key, t in r['tracker'].items():
        print(key, json.dumps({k: t[k] for k in ('true_notes', 'answered', 'missed', 'split', 'extra_answered', 'pc_right',
                                                 'pitch_right', 'heard_minus_true', 'detect_ms', 'onset_err_ms')}))
songs = r.get('songs', []) + ([r['twinkle_song']] if 'twinkle_song' in r else [])
n = 0
for s in songs:
    for row in s['tracker_rows']:
        non, dur, m, iv, na, hm, lag, raw, snap = row
        if hm is None or hm != m or na != 1:
            n += 1
            if n <= mx:
                print(f"{s['id']:18s} beat {non:6.2f} dur {dur:4.2f} true {m} heard {hm} iv_prev {iv} answered {na} "
                      f"lag {lag} raw_on {raw} snap {snap}")
print('rows shown', min(n, mx), 'of', n)
if 'songs' in r:
    for s in r['songs'][:3]:
        print(s['id'], 'voice_facts', s['voice_facts'], 'pickup', s['pickup'])
