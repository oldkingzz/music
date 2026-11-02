"""Why the live tracker mishears some clean-voice notes (sanity check (a) of round 3 phase 1). Renders one val chorale
with voice_v1 exactly as live_bench_rt does (wired, count-in 4, 90 bpm), runs RTDuet with a silent null system, and
prints the tracker's frames (time, voiced, raw YIN pitch, median pitch, aperiodicity, dB, novelty) and the segmenter's
answered notes around the given beats. Val only, read-only (prints).

  cd $ABS && .venv/bin/python runs_mini/v5rt_1101/tools/LB_diag_clean_v1.py <song index in val bach> <voice> <beat> [...]
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (installs the no-sound tripwire first)
import numpy as np  # noqa: E402
import voice_v1 as VV  # noqa: E402
from dsp import yin  # noqa: E402


class Null:
    def start(self, meta):
        pass

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return None


idx, voice, beats = int(sys.argv[1]), sys.argv[2], [float(b) for b in sys.argv[3:]]
song = LBR.bench3.eval_sets('val')['bach'][idx]
lat = LBR.LAT['wired']
T_b = 60.0 / LBR.BPM
facts = {'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': LBR.BPM}
eng = LBR.make_engine(Null(), 'null', facts, LBR.COUNT_IN)
t0h = eng.t0
y, her, vf = VV.render(song, voice, t0h, T_b)
x = np.concatenate([np.zeros(int(round(lat['Li'] * LBR.SR))), y.astype(np.float64), np.zeros(LBR.SR)])
ap_log = []
orig = eng.ft.step


def step(chunk):
    i = orig(chunk)
    ap_log.append(yin(eng.ft.buf)[1])
    return i


eng.ft.step = step
LBR.run_engine(eng, x, lat)
ft_t = np.array(eng.ft_t)
print(song['id'], voice, 'facts', vf)
for bt in beats:
    k = min(range(len(her)), key=lambda j: abs(her[j][3] - bt))
    on, off, m = her[k][0], her[k][1], her[k][2]
    print(f'--- beat {her[k][3]}: true {m} on {on:.3f} s (prev {her[k - 1][2]} off {her[k - 1][1]:.3f} s)')
    for i in np.where((ft_t > on - 0.08) & (ft_t < on + 0.08))[0]:
        print(f"  t {ft_t[i] - on:+.3f}  v {int(eng.ft.v[i])}  raw {eng.ft.raw[i]:6.2f}  m {eng.ft.m[i]:6.2f}  "
              f"ap {ap_log[i]:.3f}  db {eng.ft.db[i]:6.1f}  nov {eng.ft.nov[i]:.2f}")
    for j, (c0, jj) in enumerate(sorted(eng.by_start.items())):
        if abs(c0 - on) < 0.1:
            print(f'  answered note {jj}: segment start {c0 - on:+.3f} s, heard {eng.heard_notes[jj][2]}, '
                  f'event {eng.events[jj]}')
