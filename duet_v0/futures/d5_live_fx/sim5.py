"""A5.2 / A5.3: closed-loop room simulation  mic -> engine -> speaker -> room -> mic.

Geometry: her mouth d_m = 0.2 m from the mic, speaker 1 m from the mic, room V = 40 m^3, RT60 0.3-0.6 s
(diffuse-field model of echo/sim.py room_ir, copied below). 'Normal singing volume': Duet's direct level
at a listener 2 m from the speaker equals her direct level at 2 m  =>  speaker pressure at 1 m per unit
digital output g_spk = rms(her at 1 m) / rms(engine output), both over her singing frames (measured on an
open-loop run). The loop coefficient is then  echo_at_mic = k_mic * g_spk * (h_spk * u),  h_spk with a
direct tap of 1/1 m, plus system delay (USB/ALSA 2 x 2 periods of 64 @ 48 kHz = 5.33 ms).

    .venv/bin/python futures/d5_live_fx/sim5.py --quick     (60 s, a few points)
    .venv/bin/python futures/d5_live_fx/sim5.py             (3 min runs + gain sweep)
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fx  # noqa: E402

SR, B = 24000, 64
C_SOUND = 343.0
VOC = ROOT / 'data' / 'external' / 'vocadito'
OUT = HERE / 'results'
D_MOUTH, D_SPK, ROOM_V = 0.2, 1.0, 40.0
SYS_MS = 5.333
SING_DBFS = -20.0           # her singing level at the mic (RMS over sung frames)
NOISE_DB = -45.0            # room noise re her singing level at the mic
FEMALE = [2, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 20, 22, 24, 25, 29, 31, 32, 34, 37, 38, 39, 40]


# ---------------------------------------------------------------- acoustics (room_ir copied from echo/sim.py)
def room_ir(acoustic_ms=2.0, rt60=0.3, room_m3=40.0, q=2.0, sys_ms=20.0, hp_hz=250.0, seed=0, length_s=None):
    rng = np.random.default_rng(seed)
    length_s = length_s or rt60
    n = int((sys_ms + acoustic_ms) / 1000 * SR) + int(length_s * SR)
    h = np.zeros(n)
    d0 = (sys_ms + acoustic_ms) / 1000 * SR
    k0 = int(np.floor(d0))
    frac = d0 - k0
    r = acoustic_ms / 1000 * C_SOUND
    h[k0] += (1 - frac) / max(r, 0.05)
    h[k0 + 1] += frac / max(r, 0.05)
    direct_e = float(np.sum(h ** 2))
    r_c = 0.057 * np.sqrt(q * room_m3 / rt60)
    drr_db = 20 * np.log10(r_c / max(r, 0.05))
    t0 = k0 + int(0.0015 * SR)
    tt = np.arange(n - t0) / SR
    tail = rng.standard_normal(n - t0) * np.exp(-6.9078 * tt / rt60)
    tail *= np.sqrt(direct_e / 10 ** (drr_db / 10) / np.sum(tail ** 2))
    h[t0:] += tail
    if hp_hz:
        b, a = butter(2, hp_hz / (SR / 2), 'high')
        h = lfilter(b, a, h)
    return h, {'drr_db': float(drr_db), 'distance_m': float(r), 'critical_distance_m': float(r_c)}


class Upols:
    """Uniformly partitioned overlap-save FIR, block B. h[:B] must be zero (the loop needs >= 1 block delay)."""

    def __init__(self, h):
        P = int(np.ceil(len(h) / B))
        hp = np.zeros(P * B)
        hp[:len(h)] = h
        assert not np.any(hp[:B]), 'echo path must be delayed by at least one block'
        self.H = np.fft.rfft(hp.reshape(P, B), 2 * B, axis=1)[1:]
        self.X = np.zeros((P - 1, B + 1), complex)
        self.prev = np.zeros(B)

    def out(self):
        return np.fft.irfft(np.sum(self.H * self.X, axis=0), 2 * B)[B:]

    def push(self, x):
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = np.fft.rfft(np.concatenate([self.prev, x]))
        self.prev = x


def pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    y = np.fft.irfft(spec * (f / 1000) ** -0.5, n)
    return y / np.sqrt(np.mean(y * y))


# ---------------------------------------------------------------- her material
def load_excerpt(i):
    x, sr = sf.read(VOC / 'Audio' / f'vocadito_{i}.wav')
    x = resample_poly(x, 160, 294) if sr == 44100 else x
    notes = []
    with open(VOC / 'Annotations' / 'Notes' / f'vocadito_{i}_notesA1.csv') as fh:
        for row in csv.reader(fh):
            on, f0, dur = map(float, row[:3])
            notes.append((on, on + dur, float(fx.hz_to_midi(f0))))
    m = np.array([n[2] for n in notes])
    w = np.array([n[1] - n[0] for n in notes])
    tun = float(np.angle(np.sum(w * np.exp(2j * np.pi * m))) / (2 * np.pi))     # cents offset / 100
    pcs = np.round(m - tun).astype(int) % 12
    score = [sum(w[k] for k in range(len(m)) if (pcs[k] - t) % 12 in fx.MAJOR) for t in range(12)]
    tonic = int(np.argmax(score))
    return x, notes, tonic, tun


def session(total_s=180.0, seed=0, gap=(1.5, 3.0), ids=None):
    """Concatenate female vocadito excerpts (whole files) with silent gaps up to total_s seconds."""
    rng = np.random.default_rng(seed)
    ids = ids or FEMALE
    parts, segs, t = [np.zeros(int(1.0 * SR))], [], 1.0
    for i in ids:
        x, notes, tonic, tun = load_excerpt(i)
        fr = np.sqrt(np.mean(x[:len(x) // 240 * 240].reshape(-1, 240) ** 2, axis=1))
        lvl = np.sqrt(np.mean(fr[fr > 0.25 * fr.max()] ** 2))
        x = x / lvl * 0.1                                  # equal singing level per excerpt
        segs.append({'id': i, 't0': t, 't1': t + len(x) / SR, 'tonic': tonic, 'tuning': tun,
                     'notes': [(a + t, b + t, m) for a, b, m in notes]})
        parts.append(x)
        g = rng.uniform(*gap)
        parts.append(np.zeros(int(g * SR)))
        t += len(x) / SR + g
        if t >= total_s:
            break
    y = np.concatenate(parts)
    return y[:int(max(total_s, 1) * SR) + int(2 * SR)], segs


def frame_db(x, n=240):
    m = len(x) // n * n
    return 10 * np.log10(np.mean(x[:m].reshape(-1, n) ** 2, axis=1) + 1e-14)


# ---------------------------------------------------------------- one closed-loop run
def run(dry, segs, rt60=0.6, extra_db=0.0, guard=True, gate=True, noise_db=NOISE_DB, seed=1, open_loop=False,
        g_spk=None, return_signals=False):
    rng = np.random.default_rng(seed)
    h_her, _ = room_ir(acoustic_ms=1000 * D_MOUTH / C_SOUND, rt60=rt60, room_m3=ROOM_V, q=1.0, sys_ms=SYS_MS / 2,
                       hp_hz=0, seed=seed + 10)
    h_spk, info = room_ir(acoustic_ms=1000 * D_SPK / C_SOUND, rt60=rt60, room_m3=ROOM_V, q=2.0, sys_ms=SYS_MS,
                          hp_hz=150, seed=seed + 20)
    her_direct = dry / D_MOUTH
    her_mic = fftconvolve(dry, h_her)[:len(dry)]
    sung = frame_db(her_direct) > frame_db(her_direct).max() - 30
    lvl = np.sqrt(np.mean(np.repeat(10 ** (frame_db(her_mic) / 10), 1)[sung]))
    k_mic = 10 ** (SING_DBFS / 20) / lvl
    her_mic = her_mic * k_mic
    n = len(her_mic) // B * B
    noise = 10 ** ((SING_DBFS + noise_db) / 20) * (pink(n, rng) + 0.3 * np.sin(2 * np.pi * 120 * np.arange(n) / SR))
    mic_free = her_mic[:n] + noise
    if g_spk is None:                                       # calibrate 'normal singing volume' open loop
        u0, _ = drive(mic_free, segs, None, 0.0, guard=False, gate=gate)
        fu = frame_db(u0)
        su = sung[:len(fu)]
        rms_u = np.sqrt(np.mean(10 ** (fu[su] / 10)))
        rms_s = np.sqrt(np.mean(10 ** (frame_db(dry)[:len(fu)][su] / 10)))    # her at 1 m (digital units)
        g_spk = rms_s / rms_u
    G = k_mic * g_spk * 10 ** (extra_db / 20)
    if open_loop:
        G = 0.0
    calib = None
    if guard:
        ref = 10 ** (-35 / 20) * pink(int(2.0 * SR), rng)
        calib = (G * fftconvolve(ref, h_spk)[:len(ref)] + 10 ** ((SING_DBFS + noise_db) / 20) * pink(len(ref), rng),
                 ref)
    u, e = drive(mic_free, segs, Upols(h_spk) if G > 0 else None, G, guard=guard, gate=gate, calib=calib)
    res = {'rt60': rt60, 'extra_db': extra_db, 'guard': guard, 'gate': gate, 'g_spk': g_spk, 'G': G,
           'drr_spk_db': info['drr_db'], 'erle_soundcheck_db': e.get('erle'), 'log': e['log']}
    if return_signals:
        res['u'], res['mic'], res['her_mic'] = u, e['mic'], her_mic[:n]
    return res


def drive(mic_free, segs, path, G, guard=True, gate=True, calib=None):
    n = len(mic_free) // B * B
    eng = fx.Harmonizer(tonic=segs[0]['tonic'], aec=guard, guard=guard, log=True)
    if not gate:
        eng.open_snr = -200.0
    extra = {}
    if calib is not None:
        extra['erle'] = eng.calibrate(*calib)
    starts = sorted((int(s['t0'] * SR) // B, s['tonic']) for s in segs)
    si = 0
    u = np.zeros(n)
    mic = np.zeros(n)
    for k in range(n // B):
        while si < len(starts) and starts[si][0] <= k:
            eng.tonic = starts[si][1]                       # she chooses the key before each song
            si += 1
        echo = G * path.out() if path is not None else 0.0
        m = mic_free[k * B:(k + 1) * B] + echo
        mic[k * B:(k + 1) * B] = m
        y = eng.process(m)
        if not gate:
            pass
        u[k * B:(k + 1) * B] = y
        if path is not None:
            path.push(y)
    extra['log'] = np.array(eng.log)
    extra['mic'] = mic
    return u, extra


# ---------------------------------------------------------------- detectors
def stft_db(x, nfft=4096, hop=480):
    w = np.hanning(nfft)
    nfr = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * w, axis=1)) ** 2
    return 10 * np.log10(S + 1e-14), hop / SR


def howl(u_cl, u_ol, min_s=0.5, rise_db=10.0):
    """Howling = (a) a narrowband peak of the closed-loop output that stands > rise_db above its frame
    median AND > rise_db above the open-loop (no echo) output at the same bin AND within 3 dB of (or
    above) the strongest bin of the open-loop frame, for > min_s; or
    (b) the output level running away: closed-loop 50 ms level > open-loop + rise_db (and > -40 dBFS)
    for > min_s."""
    n = min(len(u_cl), len(u_ol))
    Sc, dt = stft_db(u_cl[:n])
    So, _ = stft_db(u_ol[:n])
    med = np.median(Sc, axis=1, keepdims=True)
    top_o = np.max(So, axis=1, keepdims=True)
    # a peak that stands out of the frame, is far above what the same engine plays without the room
    # loop, and is at least as loud as the strongest component of that no-loop output (it dominates)
    flag = (Sc > med + rise_db) & (Sc > So + rise_db) & (Sc > top_o - 3.0) & (Sc > -100)
    flag[:, :int(100 / (SR / 4096))] = False                # below 100 Hz (hum) ignored
    flag = flag | np.roll(flag, 1, axis=1) | np.roll(flag, -1, axis=1)   # +-1 bin drift
    best_nb = 0
    for j in range(flag.shape[1]):
        col = flag[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(int), [0])))
        runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
        best_nb = max(best_nb, int(runs.max()))
    lc = frame_db(u_cl[:n], 1200)
    lo = frame_db(u_ol[:n], 1200)
    run_mask = (lc > lo + rise_db) & (lc > -40)
    d = np.diff(np.concatenate(([0], run_mask.astype(int), [0])))
    runs = np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]
    best_lvl = int(runs.max()) if runs.size else 0
    return {'narrowband_longest_s': best_nb * dt, 'runaway_longest_s': best_lvl * 0.05,
            'howl': bool(best_nb * dt > min_s or best_lvl * 0.05 > min_s)}


def gate_metrics(u, dry, log, rel_db=30.0, abs_db=-40.0, segs=None):
    """A5.3. Her offsets from the DRY voice (before the room): end of a sung stretch followed by >= 0.3 s
    of silence. Time until the speaker's 5 ms level stays below -40 dBFS, and below its singing level - 30 dB.
    False triggers: gate-open events and output above -40 dBFS in her silent gaps (from offset + 0.2 s to the
    next onset)."""
    fd = frame_db(dry, 120)                                # 5 ms frames
    sing = np.median(fd[fd > np.max(fd) - 35])            # her typical singing level (dry)
    act = fd > sing - 20                                   # she sings while within 20 dB of it
    fu = frame_db(u, 120)
    m = min(len(fd), len(fu))
    act, fd, fu = act[:m], fd[:m], fu[:m]
    sing_u = 10 * np.log10(np.mean(10 ** (fu[act] / 10)))
    thr_rel = sing_u - rel_db
    d = np.diff(act.astype(int))
    offs = np.nonzero(d == -1)[0] + 1
    ons = np.nonzero(d == 1)[0] + 1
    rel_abs, rel_rel, gaps = [], [], []
    for o in offs:
        nxt = ons[ons > o]
        end = nxt[0] if nxt.size else m
        if end - o < int(0.3 / 0.005):
            continue
        seg = fu[o:end]
        t_abs = next((i for i in range(len(seg)) if np.all(seg[i:] < abs_db)), len(seg))
        t_rel = next((i for i in range(len(seg)) if np.all(seg[i:] < thr_rel)), len(seg))
        rel_abs.append(t_abs * 0.005)
        rel_rel.append(t_rel * 0.005)
        gaps.append((o, end))
    # gate log is per engine block (64 samples)
    gopen = log[:, 5].astype(int)
    fa_events, fa_time, gap_time, loud_time = 0, 0.0, 0.0, 0.0
    for o, end in gaps:
        a = int((o * 120 + 0.2 * SR) // B)
        b = int((end * 120 - 0.02 * SR) // B)
        if b <= a:
            continue
        g = gopen[a:b]
        fa_events += int(np.sum(np.diff(g) == 1)) + int(g[0] == 1 and gopen[a - 1] == 0)
        fa_time += float(np.sum(g)) * B / SR
        gap_time += (b - a) * B / SR
        uu = fu[o + 40:end - 4]
        loud_time += float(np.sum(uu > abs_db)) * 0.005
    # the digital silences between songs (her dry signal is exactly zero there)
    sb_events, sb_open, sb_time, sb_loud = 0, 0.0, 0.0, 0.0
    for s0, s1 in zip(segs[:-1], segs[1:]) if segs else []:
        a = int((s0['t1'] + 0.2) * SR) // B
        b = int((s1['t0'] - 0.02) * SR) // B
        if b <= a or b > len(gopen):
            continue
        g = gopen[a:b]
        sb_events += int(np.sum(np.diff(g) == 1)) + int(g[0] == 1 and gopen[a - 1] == 0)
        sb_open += float(np.sum(g)) * B / SR
        sb_time += (b - a) * B / SR
        uu = fu[int((s0['t1'] + 0.2) / 0.005):int((s1['t0'] - 0.02) / 0.005)]
        sb_loud += float(np.sum(uu > abs_db)) * 0.005
    ra, rr = np.array(rel_abs), np.array(rel_rel)
    return {'n_offsets': int(len(ra)), 'singing_level_out_dbfs': float(sing_u),
            'release_to_-40dBFS_s': {'median': float(np.median(ra)), 'p95': float(np.percentile(ra, 95)),
                                     'max': float(ra.max()), 'share_le_0.2': float(np.mean(ra <= 0.2))},
            'release_to_sing-30dB_s': {'median': float(np.median(rr)), 'p95': float(np.percentile(rr, 95)),
                                       'max': float(rr.max()), 'share_le_0.2': float(np.mean(rr <= 0.2))},
            'gap_time_s': gap_time, 'false_trigger_events': fa_events, 'false_open_time_s': fa_time,
            'gap_output_above_-40dBFS_s': loud_time,
            'between_songs': {'time_s': sb_time, 'gate_open_events': sb_events, 'gate_open_s': sb_open,
                              'output_above_-40dBFS_s': sb_loud}}


# ---------------------------------------------------------------- experiments
_SESSION = {}


def _job(args):
    total_s, rt60, extra_db, guard, gate, g_spk = args
    if total_s not in _SESSION:
        _SESSION[total_s] = session(total_s)
    dry, segs = _SESSION[total_s]
    t0 = time.time()
    ol = run(dry, segs, rt60=rt60, guard=guard, gate=gate, open_loop=True, g_spk=g_spk, return_signals=True)
    cl = run(dry, segs, rt60=rt60, extra_db=extra_db, guard=guard, gate=gate, g_spk=ol['g_spk'],
             return_signals=True)
    h = howl(cl['u'], ol['u'])
    gm = gate_metrics(cl['u'], dry[:len(cl['u'])], cl['log'], segs=segs)
    gm_ol = gate_metrics(ol['u'], dry[:len(ol['u'])], ol['log'], segs=segs)   # same engine, no room loop
    peak_out = float(20 * np.log10(np.max(np.abs(cl['u'])) + 1e-12))
    return {'total_s': total_s, 'rt60': rt60, 'extra_db': extra_db, 'guard': guard, 'gate': gate,
            'howl': h, 'gate_metrics': gm, 'gate_metrics_open_loop': gm_ol, 'erle_soundcheck_db': cl['erle_soundcheck_db'],
            'peak_out_dbfs': peak_out, 'g_spk': ol['g_spk'], 'secs': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--procs', type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.quick:
        jobs = [(60.0, 0.6, x, g, True, None) for g in (True, False) for x in (0.0, 12.0)]
    else:
        jobs = [(180.0, r, 0.0, g, True, None) for r in (0.3, 0.6) for g in (True, False)]
        jobs += [(60.0, r, x, g, True, None) for r in (0.3, 0.6) for g in (True, False)
                 for x in (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 30.0)]
    with Pool(a.procs) as p:
        res = p.map(_job, jobs)
    name = 'sim_quick.json' if a.quick else 'sim.json'
    (OUT / name).write_text(json.dumps(res, indent=1, default=float))
    for r in res:
        print(r['total_s'], r['rt60'], r['extra_db'], 'guard' if r['guard'] else 'noguard', r['howl'],
              'rel40', round(r['gate_metrics']['release_to_-40dBFS_s']['p95'], 3),
              'rel30', round(r['gate_metrics']['release_to_sing-30dB_s']['p95'], 3),
              'fa', r['gate_metrics']['false_trigger_events'], round(r['gate_metrics']['gap_output_above_-40dBFS_s'], 2),
              'erle', r['erle_soundcheck_db'], 'peak', round(r['peak_out_dbfs'], 1), round(r['secs'], 1))


if __name__ == '__main__':
    main()
