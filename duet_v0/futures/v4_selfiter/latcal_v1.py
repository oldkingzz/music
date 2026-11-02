"""latcal_v1: the Duet live page's round-trip latency calibration ('测延迟'), pure DSP (run v5rt_1101, round 4).

Nothing here opens a sound device or plays anything. The page (futures/v4_selfiter/live_panel_v3.py) plays the clicks
through its own duplex stream callback (the one the songs use) and hands the microphone recording to estimate();
make_test() builds synthetic recordings for the offline tests (runs_mini/v5rt_1101/tools/latcal_selftest_v1.py).

The step (schedule()): after LEAD s of silence, COUNT_IN count-in clicks (she listens), then N_CLICKS higher clicks at
BPM (100 bpm: 0.6 s apart); she sings a short 'da' (or 'la') on each of those, in any register. The recording is the
input of the same stream, sample-aligned with the output: input block k arrives in the same callback as output block k.
A click written at output sample c is heard Lo_true later, she sings on it, her voice reaches the input Li_true later,
so it starts at input sample c + (Lo_true + Li_true) * sr, plus her own timing error. So

    RT = median over the responses found of (onset - click write sample) / sr

is the ROUND TRIP of live_bench_rt_v2's latency model, and the page tells the engine Li = max(0, RT - Lo) with Lo the
stream's output latency: the round-trip error of every heard onset is then about 0 (whatever she does systematically -
singing a little ahead of the click, say - is in the song too, so it is calibrated away with the device latency).

estimate(rec, sr, click_write_samples) -> dict(rt_s, n_used, spread_s, ok, why, ...):
 1. A 60 Hz high-pass (2nd-order Butterworth, causal, so nothing is ever moved earlier), then an energy envelope in dB
    (5 ms RMS windows every 2.5 ms). Noise floor F = 20th percentile of the envelope before the first response click
    (lead and count-in, where she is silent), or of the whole recording when that is too short.
 2. Candidate onsets: the envelope rises >= 12 dB above its minimum over the previous 60 ms and is >= F + 12 dB (from
    silence, or from the dip of a 'd' / 'l' between two vowels), or rises >= 12 dB within 20 ms. The onset is where that
    rise starts: the last envelope frame within 6 dB of the minimum, then refined to the first sample whose 2 ms power
    exceeds 6x the noise power of the frames before it (a 'd' burst or the first glottal pulses; never the breath before).
 3. A candidate counts as her response only if, from its onset: the envelope peaks >= F + 15 dB and >= -65 dBFS within
    150 ms (not room noise), stays >= F + 12 dB for >= 60 ms of the next 250 ms (not a click leaking in, not a tap),
    and is voiced (YIN's cumulative mean normalized difference < 0.35 for some f0 in 60..1000 Hz, on 60 ms from 30 ms
    after the onset: a sung vowel in any register; not a breath, a cough or a door). Candidates within 100 ms after an
    accepted onset are dropped (the vowel after the 'd', a scoop).
 4. Matching: RT* = the round trip in -50..550 ms (1 ms grid) that the most clicks have a response within +-60 ms of
    (ties: the smallest total deviation); each click keeps its response nearest to click + RT*. A missed click, an
    early or late response (more than 60 ms off) and extra sounds are simply not used.
 5. RT = median of the used (onset - click) / sr; spread = their median absolute deviation from RT. ok only if at least
    MIN_FOUND (5) of the N_CLICKS (8) responses are used and spread <= 40 ms; otherwise ok = False and `why` says, in
    short Chinese, what to do (sing louder / shorter / on the click, then retry). Also refused: click 1 has no response
    but a voiced sound comes after the last response (the round trip may exceed 0.55 s, beyond one click interval).
make_test(rt_s, seed, ...) -> dict(rec, sr, clicks, truth, ...): the schedule's clicks and her synthetic 'da' on each
response click at click + rt_s + N(0, 15 ms): a short noise burst for the 'd' (8-15 ms, 12-20 dB below the vowel), a
voice-onset gap of 4-15 ms, then a short sung vowel rendered like voice_v1 / singer.render (harmonics through singer's
'ah' formants, 30-70 ms raised-cosine attack, 50 ms release, per-note detune N(0, 0.15 st), half of the notes scooped
from 50-150 cents flat over 60-120 ms, vibrato growing from 0.12 s, breathiness), Bach-style stepwise pitches in a random
major key around MIDI 67 (or 50 with low=True), in-breaths before the first and about 40 % of the other responses, and
pink room noise 40 dB below her voice's RMS (voice at -30 dBFS +- 6 dB RMS). Options: missing=j (no response to click
j), early=j (response j comes 120 ms early), respond=False (room noise and breaths only), bleed=True (the clicks leak
into the microphone 30 dB below her voice level, as with loudspeakers), silent=True (all zeros).
"""
import numpy as np

SR = 24000                     # the page's sample rate (dsp.SR); estimate() takes sr explicitly
BPM = 100.0
COUNT_IN = 4
N_CLICKS = 8
LEAD = 0.5                     # s of silence before the first count-in click (stream start-up, noise floor)
TAIL = 1.3                     # s after the last response click (round trips up to 0.55 s plus her 'da')
CLICK_S = 0.025                # the click: RTDuet._click's 25 ms decaying sine

ENV_HOP, ENV_WIN = 0.0025, 0.005
HP_HZ = 60.0
RT_MIN, RT_MAX = -0.05, 0.55   # s: round trips searched (she may sing a little ahead of the click)
MATCH = 0.06                   # s: a response belongs to a click if within this of click + RT*
MIN_FOUND = 5
MAX_SPREAD = 0.040
RISE_DB, RISE_LOOK, START_DB = 12.0, 0.06, 6.0
SHARP_LOOK = 0.02
PEAK_DB, ABS_DB, PEAK_WIN = 15.0, -65.0, 0.15
DUR_MIN, DUR_WIN = 0.06, 0.25
VOICE_A, VOICE_N = 0.03, 0.06
F0_MIN, F0_MAX = 60.0, 1000.0
CMND_MAX = 0.35
DEDUP = 0.10
FINE_S, FINE_X = 0.002, 6.0
FORMANTS_AH = [(730, 90, 1.0), (1090, 110, 0.5), (2440, 170, 0.25)]   # singer.FORMANTS_AH


# ---------------------------------------------------------------------------------------------------- the clicks
def schedule(sr=SR, bpm=BPM, count_in=COUNT_IN, n_clicks=N_CLICKS, lead=LEAD, tail=TAIL):
    """Output sample at which each click is written: count_in (listen) and clicks (sing 'da' on each)."""
    T = 60.0 / bpm
    cin = [int(round((lead + k * T) * sr)) for k in range(count_in)]
    clicks = [int(round((lead + (count_in + j) * T) * sr)) for j in range(n_clicks)]
    return {'sr': sr, 'bpm': bpm, 'count_in': cin, 'clicks': clicks, 'n_total': clicks[-1] + int(round(tail * sr))}


def click_wave(kind, sr=SR):
    """RTDuet._click's sound: count-in 1320 Hz at 0.22, response clicks 1760 Hz at 0.35 (higher, louder)."""
    n = int(CLICK_S * sr)
    t = np.arange(n) / sr
    f, a = (1320.0, 0.22) if kind == 'count' else (1760.0, 0.35)
    return a * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.006)


def render_clicks(sched, n0, n):
    """The clicks in output samples [n0, n0 + n)."""
    sr = sched['sr']
    y = np.zeros(n)
    for kind, starts in (('count', sched['count_in']), ('resp', sched['clicks'])):
        w = click_wave(kind, sr)
        for i0 in starts:
            a, b = max(i0, n0), min(i0 + len(w), n0 + n)
            if b > a:
                y[a - n0:b - n0] += w[a - i0:b - i0]
    return y


# ---------------------------------------------------------------------------------------------------- the estimate
def _highpass(x, sr):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, HP_HZ, 'highpass', fs=sr, output='sos'), x)


def _envelope(x, sr):
    H, W = int(round(ENV_HOP * sr)), int(round(ENV_WIN * sr))
    n = (len(x) - W) // H + 1
    if n <= 0:
        return np.zeros(0), H, W
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    idx = np.arange(n) * H
    p = (c[idx + W] - c[idx]) / W
    return 10.0 * np.log10(np.maximum(p, 1e-12)), H, W


def _cmnd_min(seg, sr):
    """YIN's cumulative mean normalized difference, its minimum over lags for f0 in F0_MIN..F0_MAX (1 = noise)."""
    tmin, tmax = int(sr / F0_MAX), int(sr / F0_MIN)
    N = len(seg) - tmax
    if N < tmax or not np.any(seg):
        return 1.0
    x0 = seg[:N]
    nfft = 1 << int(np.ceil(np.log2(len(seg) + N)))
    R = np.fft.irfft(np.fft.rfft(seg, nfft) * np.conj(np.fft.rfft(x0, nfft)), nfft)
    taus = np.arange(1, tmax + 1)
    c = np.concatenate([[0.0], np.cumsum(seg * seg)])
    d = float(np.dot(x0, x0)) + (c[taus + N] - c[taus]) - 2.0 * R[taus]
    d = np.maximum(d, 0.0)
    cm = d * taus / np.maximum(np.cumsum(d), 1e-30)
    return float(np.min(cm[tmin - 1:]))


def _onsets(x, sr, floor_end):
    """Her voiced onsets (input sample indices) and the facts used to accept them."""
    E, H, W = _envelope(x, sr)
    n = len(E)
    if n == 0:
        return [], {'floor_db': None, 'candidates': 0}
    pre = E[:max(0, (floor_end - W) // H)]
    F = float(np.percentile(pre, 20)) if len(pre) >= int(0.3 / ENV_HOP) else float(np.percentile(E, 10))
    L, S = int(round(RISE_LOOK / ENV_HOP)), int(round(SHARP_LOOK / ENV_HOP))
    p2 = np.convolve(x * x, np.ones(int(FINE_S * sr)) / int(FINE_S * sr))[:len(x)]   # 2 ms trailing power
    pk_n, dur_n = int(PEAK_WIN / ENV_HOP), int(DUR_WIN / ENV_HOP)
    a_v, n_v, tmax = int(VOICE_A * sr), int(VOICE_N * sr), int(sr / F0_MIN)
    found, why_not, prev, n_cand = [], [], False, 0
    for k in range(L, n):
        j0 = k - L + int(np.argmin(E[k - L:k]))
        now = E[k] - E[j0] >= RISE_DB and E[k] >= F + RISE_DB
        sharp = E[k] >= F + RISE_DB and E[k] - float(np.min(E[k - S:k])) >= RISE_DB
        new = (now and not prev) or sharp
        prev = now
        if not new:
            continue
        js = j0 + int(np.nonzero(E[j0:k] <= E[j0] + START_DB)[0][-1])
        a0 = js * H
        if found and a0 < found[-1] + DEDUP * sr:
            continue
        n_cand += 1
        pn = float(np.mean(10.0 ** (E[max(k - L, 0):js + 1] / 10.0)))
        hit = np.nonzero(p2[a0:(k * H + W)] >= FINE_X * pn)[0]
        s0 = a0 + int(hit[0]) if len(hit) else (js + 1) * H + W - H
        if found and s0 < found[-1] + DEDUP * sr:
            continue
        f0 = s0 // H
        peak = float(np.max(E[f0:f0 + pk_n])) if f0 < n else -999.0
        if peak < F + PEAK_DB or peak < ABS_DB:
            why_not.append('quiet')
            continue
        if np.sum(E[f0:f0 + dur_n] >= F + RISE_DB) * ENV_HOP < DUR_MIN:
            why_not.append('short')
            continue
        seg = x[s0 + a_v:s0 + a_v + n_v + tmax]
        if _cmnd_min(seg, sr) > CMND_MAX:
            why_not.append('unvoiced')
            continue
        found.append(s0)
    return found, {'floor_db': round(F, 1), 'candidates': n_cand,
                   'rejected': {w: why_not.count(w) for w in sorted(set(why_not))}}


def estimate(rec, sr, click_write_samples, min_found=MIN_FOUND, max_spread=MAX_SPREAD):
    """Round trip from a calibration recording (see the module docstring). rec: microphone input aligned with the
    output stream (rec[i] arrived in the callback that wrote output sample i); click_write_samples: the output samples
    at which the response clicks were written. Returns dict(rt_s, n_used, spread_s, ok, why, offsets_ms, ...)."""
    x = np.asarray(rec, np.float64).ravel()
    clicks = np.asarray(click_write_samples, np.int64)
    out = {'rt_s': None, 'n_used': 0, 'spread_s': None, 'ok': False, 'why': '', 'n_clicks': int(len(clicks)),
           'offsets_ms': [None] * int(len(clicks)), 'n_onsets': 0}
    if len(x) < int(0.5 * sr) or not len(clicks):
        out['why'] = '录音太短，没测完：再测一次。'
        return out
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)
    xh = _highpass(x, sr)
    on, facts = _onsets(xh, sr, int(clicks[0] + RT_MIN * sr))
    out.update(facts)
    out['n_onsets'] = len(on)
    if not on:
        out['why'] = '没听到你唱：看电平条有没有动，跟着高的嘀声每下唱一个短“哒”，再测一次。'
        return out
    on = np.asarray(on, np.int64)
    d = (on[None, :] - clicks[:, None]) / float(sr)
    best = None
    for g in np.arange(RT_MIN, RT_MAX + 1e-9, 0.001):
        m = np.min(np.abs(d - g), axis=1)
        hit = m <= MATCH
        score = (int(hit.sum()), -float(m[hit].sum()))
        if best is None or score > best[0]:
            best = (score, float(g))
    g = best[1]
    used, used_on = [], set()
    for j in range(len(clicks)):
        i = int(np.argmin(np.abs(d[j] - g)))
        if abs(d[j, i] - g) <= MATCH:
            used.append(float(d[j, i]))
            used_on.add(i)
            out['offsets_ms'][j] = round(1000.0 * float(d[j, i]), 1)
    out['n_used'] = len(used)
    if not used:
        out['why'] = '没对上嘀声：跟着高的嘀声每下唱一个短“哒”，再测一次。'
        return out
    rt = float(np.median(used))
    spread = float(np.median(np.abs(np.asarray(used) - rt)))
    out.update({'rt_s': round(rt, 4), 'spread_s': round(spread, 4), 'rt_grid_s': round(g, 3)})
    late = [int(i) for i in range(len(on)) if i not in used_on and on[i] > clicks[-1] + (g + MATCH) * sr]
    if len(used) < min_found:
        out['why'] = f'只对上了 {len(used)} 下（8 下里要至少 {min_found} 下）：每下唱短一点、踩在嘀声上，再测一次。'
    elif spread > max_spread:
        out['why'] = f'每下差得太多（约 {spread * 1000:.0f} 毫秒）：尽量踩在嘀声上唱，再测一次。'
    elif out['offsets_ms'][0] is None and late:
        out['why'] = '第一下没对上、最后一下之后还有声音：延迟可能超过半秒，或者多唱了。再测一次。'
    else:
        out['ok'] = True
    return out


# ---------------------------------------------------------------------------------------------------- synthetic tests
def _colored(n, rng, slope):
    """dsp.colored_noise (copied): unit-RMS noise with an amplitude slope per octave (-0.5: pink)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    spec *= (f / 1000.0) ** slope
    y = np.fft.irfft(spec, n)
    return y / (np.sqrt(np.mean(y * y)) + 1e-12)


def _formant_gain(f):
    g = np.full_like(f, 0.04)
    for fc, bw, amp in FORMANTS_AH:
        g += amp / np.sqrt(1.0 + ((f - fc) / (bw / 2.0)) ** 2)
    return g


def _da(buf, on, dur, midi, rng, sr):
    """One sung 'da' starting (its 'd' burst) at `on` seconds; added into buf. Returns the vowel's sample span."""
    i0 = int(round(on * sr))
    nb = int(rng.uniform(0.008, 0.015) * sr)
    burst = np.diff(np.concatenate([[0.0], rng.standard_normal(nb)])) * np.exp(-np.arange(nb) / (0.004 * sr))
    burst *= 10.0 ** (rng.uniform(-20.0, -12.0) / 20.0) / max(np.max(np.abs(burst)), 1e-9)
    v0 = i0 + int(rng.uniform(0.004, 0.015) * sr)
    nv = int(dur * sr)
    t = np.arange(nv) / sr
    m = midi + rng.normal(0.0, 0.15)
    pitch = np.full(nv, m)
    if rng.random() < 0.5:                                   # scoop: starts d flat, reaches the pitch after s
        dd, s = rng.uniform(0.5, 1.5), rng.uniform(0.06, 0.12)
        k = min(nv, int(s * sr))
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / max(k, 1))
        pitch[:k] = m - dd * (1.0 - w)
    vib = 0.30 * np.clip((t - 0.12) / 0.3, 0.0, 1.0) * np.sin(2 * np.pi * 5.5 * t)
    f0 = 440.0 * 2.0 ** ((pitch + vib - 69.0) / 12.0)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(nv)
    for h in range(1, 61):
        fh = h * f0
        keep = np.clip((5000.0 - fh) / 500.0, 0.0, 1.0)
        if not keep.any():
            break
        y += keep * _formant_gain(fh) / h * np.sin(h * ph)
    y += 0.012 * np.max(np.abs(y)) * _colored(nv, rng, -0.3)          # breathiness
    att = rng.uniform(0.03, 0.07)
    y *= (0.5 - 0.5 * np.cos(np.pi * np.clip(t / att, 0.0, 1.0))) * np.clip((dur - t) / 0.05, 0.0, 1.0)
    y /= max(np.max(np.abs(y)), 1e-9)
    e = min(len(buf), v0 + nv)
    if e > v0 >= 0:
        buf[v0:e] += y[:e - v0]
    e = min(len(buf), i0 + nb)
    if e > i0 >= 0:
        buf[i0:e] += burst[:e - i0]
    return v0, v0 + nv


def make_test(rt_s, seed, missing=None, early=None, low=False, respond=True, silent=False, bleed=False,
              jitter=0.015, noise_db=-40.0, sr=SR, sched=None):
    """A synthetic microphone recording of the calibration step (module docstring). Returns dict(rec float32, sr,
    clicks (response click write samples), truth (her true onset in seconds per response click, None if none),
    rt_s, offsets_true_s, level_dbfs, kind)."""
    sched = sched or schedule(sr)
    clicks = list(sched['clicks'])
    n = sched['n_total']
    code = 1 + 2 * bool(low) + 4 * (missing is not None) + 8 * (early is not None) + 16 * (not respond) + 32 * bool(bleed)
    rng = np.random.default_rng([int(seed), int(round(rt_s * 1e4)), code])
    kind = 'silent' if silent else ('no_response' if not respond else 'response')
    if silent:
        return {'rec': np.zeros(n, np.float32), 'sr': sr, 'clicks': clicks, 'truth': [None] * len(clicks),
                'rt_s': rt_s, 'offsets_true_s': [], 'level_dbfs': None, 'kind': kind}
    level = 0.03 * 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)          # her voice RMS during her notes
    voice = np.zeros(n)
    tonic = int(rng.integers(0, 12))
    scale = [0, 2, 4, 5, 7, 9, 11]
    center = 50 if low else 67
    deg = int(rng.integers(0, 7))
    jit = rng.normal(0.0, jitter, len(clicks))
    truth, spans = [], []
    for j, c in enumerate(clicks):
        deg += int(rng.choice([-2, -1, -1, 0, 1, 1, 2]))
        octv, dg = divmod(deg, 7)
        midi = tonic + 12 * octv + scale[dg]
        midi += 12 * int(np.round((center - midi) / 12.0))
        dur = rng.uniform(0.15, 0.32)
        if not respond or j == missing:
            truth.append(None)
            continue
        on = c / sr + rt_s + float(jit[j]) - (0.12 if j == early else 0.0)
        spans.append(_da(voice, on, dur, midi, rng, sr))
        truth.append(on)
    mask = np.zeros(n, bool)
    for a, b in spans:
        mask[a:b] = True
    if mask.any():
        voice *= level / max(float(np.sqrt(np.mean(voice[mask] ** 2))), 1e-12)
    breath = np.zeros(n)
    for j, on in enumerate(truth if respond else [c / sr + rt_s + float(jit[j]) for j, c in enumerate(clicks)]):
        if on is None or not (j == 0 or rng.random() < 0.4):
            continue
        a, b = int((on - 0.12) * sr), int((on - 0.02) * sr)
        if a > 0:
            breath[a:b] += level * 10.0 ** (-30.0 / 20.0) * _colored(b - a, rng, 0.2) * np.hanning(b - a)
    rec = voice + breath + _colored(n, rng, -0.5) * level * 10.0 ** (noise_db / 20.0)
    if bleed:
        w = click_wave('resp', sr) / 0.35
        for c in sched['count_in'] + clicks:
            a = c + int(round(rt_s * sr))
            e = min(n, a + len(w))
            if e > a:
                rec[a:e] += level * 10.0 ** (-30.0 / 20.0) * 4.0 * w[:e - a]
    offs = [t - c / sr for t, c in zip(truth, clicks) if t is not None]
    return {'rec': rec.astype(np.float32), 'sr': sr, 'clicks': clicks, 'truth': truth, 'rt_s': rt_s,
            'offsets_true_s': offs, 'level_dbfs': round(20.0 * np.log10(level), 1), 'kind': kind}
