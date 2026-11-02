"""A5.4 clash time. (a) symbolic on the duet_beat val songs, (b) the real engine on vocadito excerpts.

    .venv/bin/python futures/d5_live_fx/clash.py       -> results/clash.json

(a) Her TRUE notes in seconds (song bpm, else 100). Duet hears her through bench.heard (10 % of notes
    misheard, fixed seed). Two models, both with the engine's measured lags (results/latency.json):
    literal  Duet's note = diatonic third below her HEARD note; it changes L_note after her note
             changes (legato: the old harmony note holds meanwhile; after a rest: silent for L_onset).
    shifter  what the engine physically does: Duet's pitch = her CURRENT pitch shifted down by
             s = heard - third_below(heard); the audio path follows her after L_path (old pitch
             meanwhile), the shift interval s switches after L_note.
    Shares are time-weighted over all her note time (Duet joins immediately: no listening beats).
(b) Every vocadito excerpt (female ones with the 140 Hz mode, the rest with the 80 Hz male mode) goes
    through the engine at its key (estimated from the annotation = the key she would choose); the
    output's pitch is tracked with YIN every 10 ms and compared with her annotated notes, both rounded
    after removing the singer's tuning offset.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'duet_beat'))
import fx  # noqa: E402
import sim5  # noqa: E402

GOOD = {0, 3, 4, 7, 8, 9}
HARSH = {1, 2, 6, 10, 11}
SR = 24000


def shares_of(segments):
    """segments: [(dur, interval or None)] -> shares."""
    tot = sum(d for d, _ in segments) or float('nan')
    out = {'good': 0.0, 'harsh': 0.0, 'fourth': 0.0, 'silent': 0.0}
    for d, iv in segments:
        if iv is None:
            out['silent'] += d
        elif iv in GOOD:
            out['good'] += d
        elif iv in HARSH:
            out['harsh'] += d
        else:
            out['fourth'] += d
    r = {k: v / tot for k, v in out.items()}
    r['not_good'] = 1.0 - r['good']
    r['time_s'] = tot
    return r


def song_segments(song, heard, L_path, L_note, L_on, model, legato_s=0.05):
    bpm = song.get('bpm') or song.get('tempo_bpm') or 100.0
    sec = 60.0 / bpm
    tonic = song['tonic']
    segs = []
    prev = None                                   # (true midi, heard midi, off time)
    for (on, off, m), (_, _, h) in zip(song['sop'], heard['sop']):
        on, off = on * sec, off * sec
        dur = off - on
        if dur <= 0:
            continue
        s_new = h - fx.third_below(h, tonic)
        legato = prev is not None and on - prev[2] < legato_s
        if model == 'literal':
            new = fx.third_below(h, tonic)
            if legato:
                old = fx.third_below(prev[1], prev[1] % 12 * 0 + tonic)
                a = min(L_note, dur)
                segs.append((a, abs(m - old) % 12))
                segs.append((dur - a, abs(m - new) % 12))
            else:
                a = min(L_on + L_note * 0, dur)
                segs.append((a, None))
                segs.append((dur - a, abs(m - new) % 12))
        else:
            s_old = prev[1] - fx.third_below(prev[1], tonic) if prev is not None else s_new
            if legato:
                a = min(L_path, dur)
                b = min(L_note, dur)
                segs.append((a, abs(m - (prev[0] - s_old)) % 12))
                segs.append((max(b - a, 0.0), abs(m - (m - s_old)) % 12))
                segs.append((dur - max(a, b), abs(m - (m - s_new)) % 12))
            else:
                a = min(L_on, dur)
                segs.append((a, None))
                segs.append((dur - a, abs(s_new) % 12))
        prev = (m, h, off)
    return segs


def symbolic(lat):
    import bench
    L_path = max(d['delay_ms'] for d in lat['path_delay']) / 1000
    L_note = max(d['settle_ms'] for d in lat['pitch_step']) / 1000
    L_on = max(d['onset_ms'] for d in lat['onset']) / 1000
    out = {'lags_s': {'path': L_path, 'note': L_note, 'onset': L_on}}
    ev = bench.eval_sets('val')
    for model in ('literal', 'shifter'):
        rows = {}
        for ds in ('bach', 'folk', 'pop909', 'pop'):
            segs = []
            for s in ev[ds]:
                segs += song_segments(s, bench.heard(s), L_path, L_note, L_on, model)
            rows[ds] = shares_of(segs)
            rows[ds]['songs'] = len(ev[ds])
        # sensitivity: 5x the measured note lag
        rows['bach_lag_x5'] = shares_of(sum((song_segments(s, bench.heard(s), L_path * 5, L_note * 5, L_on * 5, model)
                                             for s in ev['bach']), []))
        out[model] = rows
    # self-test: no mishearing and zero lag -> literal model is all thirds (good = 1) when she sings legato
    s0 = ev['bach'][0]
    t = shares_of(song_segments(s0, s0, 0.0, 0.0, 0.0, 'literal'))
    assert abs(t['good'] - 1.0) < 1e-9, t
    return out


def track_out(y, lvl_floor):
    yin = fx.StreamYin(SR, 90, 1000)
    hop = 240
    m = np.full(len(y) // hop, np.nan)
    for j in range(len(m)):
        k = (j + 1) * hop
        if k < yin.N:
            continue
        seg = y[k - yin.N:k]
        if 10 * np.log10(np.mean(seg ** 2) + 1e-14) < lvl_floor:
            continue
        f, ap = yin(seg)
        if ap < 0.3:
            m[j] = float(fx.hz_to_midi(f))
    return m                                       # frame j covers ((j+1)*hop - N, (j+1)*hop]


def audio(ids=range(1, 41), mode='correct'):
    rows, segs_all, segs_v, segs_s, segs_u, segs_c, segs_i = [], [], [], [], [], [], []
    import csv
    meta = {int(r['track_id']): int(r['average_pitch']) for r in
            csv.DictReader(open(sim5.VOC / 'vocadito_metadata.csv'))}
    for i in ids:
        x, notes, tonic, tun = sim5.load_excerpt(i)
        fmin = 140.0 if i in sim5.FEMALE else 80.0
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
        y, eng = fx.run_open(x, tonic, aec=False, fmin=fmin, mode=mode)
        fy = sim5.frame_db(y)
        lvl = 10 * np.log10(np.mean(10 ** (fy[fy > fy.max() - 30] / 10)))
        mo = track_out(y, lvl - 30)
        fxx = sim5.frame_db(x)
        lx = 10 * np.log10(np.mean(10 ** (fxx[fxx > fxx.max() - 30] / 10)))
        mi = track_out(x, lx - 30)                     # her own pitch track (is she voiced?)
        tcen = ((np.arange(len(mo)) + 1) * 240 - fx.StreamYin(SR, 90, 1000).N / 2) / SR   # frame centre
        segs, sv, ss, su, sc, si = [], [], [], [], [], []
        for on, off, mm in notes:
            sel = np.nonzero((tcen >= on) & (tcen < off))[0]
            her = int(np.round(mm - tun))
            unamb = abs((mm - tun) - her) <= 0.3
            for j in sel:
                v = mo[j]
                iv = None if np.isnan(v) else abs(her - int(np.round(v - tun))) % 12
                segs.append((0.01, iv))                          # both rounded to her tuned semitones
                ss.append((0.01, None if np.isnan(v) else int(abs(np.round(mm - v))) % 12))   # sounding interval
                if unamb:
                    su.append((0.01, iv))
                if tcen[j] - on > 0.06 and off - tcen[j] > 0.06:
                    sc.append((0.01, iv))                        # note cores (no first/last 60 ms)
                # ideal causal-free follower: her own pitch track shifted by the right diatonic interval
                if j < len(mi):
                    hv = mi[j]
                    s_ok = her - fx.third_below(her, tonic)
                    si.append((0.01, None if np.isnan(hv) else abs(her - int(np.round(hv - s_ok - tun))) % 12))
                if j < len(mi) and not np.isnan(mi[j]):
                    sv.append((0.01, iv))
        r = shares_of(segs)
        rv = shares_of(sv)
        r['sounding'] = shares_of(ss)
        r['unambiguous_notes'] = shares_of(su)
        segs_c.extend(sc)
        segs_i.extend(si)
        segs_s.extend(ss)
        segs_u.extend(su)
        r.update({'id': i, 'tonic': tonic, 'tuning_cents': round(100 * tun), 'avg_pitch': meta[i],
                  'mode': 'female' if fmin > 100 else 'male', 'voiced_only': rv})
        rows.append(r)
        segs_all += segs
        segs_v += sv
    return {'engine_mode': mode, 'all': shares_of(segs_all), 'all_her_voiced_frames': shares_of(segs_v),
            'all_sounding_interval': shares_of(segs_s), 'all_unambiguous_notes': shares_of(segs_u),
            'all_note_cores': shares_of(segs_c), 'ideal_follower_of_her_pitch': shares_of(segs_i),
            'per_excerpt': rows,
            'female_only': shares_weighted([r for r in rows if r['mode'] == 'female']),
            'male_only': shares_weighted([r for r in rows if r['mode'] == 'male'])}


def shares_weighted(rows):
    T = sum(r['time_s'] for r in rows)
    return {k: sum(r[k] * r['time_s'] for r in rows) / T for k in ('good', 'harsh', 'fourth', 'silent', 'not_good')}


def main():
    lat = json.loads((HERE / 'results' / 'latency.json').read_text())
    from multiprocessing import Pool
    aud = [audio_mode('correct'), audio_mode('follow')]
    res = {'symbolic': symbolic(lat), 'audio': aud[0], 'audio_follow_mode': aud[1]}
    (HERE / 'results' / 'clash.json').write_text(json.dumps(res, indent=1, default=float))
    for model in ('literal', 'shifter'):
        for ds, r in res['symbolic'][model].items():
            print(model, ds, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    for key in ('audio', 'audio_follow_mode'):
        a = res[key]
        print(key, 'all', {k: round(v, 4) for k, v in a['all'].items()})
        print(key, 'her voiced', {k: round(v, 4) for k, v in a['all_her_voiced_frames'].items()})
        print(key, 'sounding', {k: round(v, 4) for k, v in a['all_sounding_interval'].items()})
        print(key, 'unambiguous', {k: round(v, 4) for k, v in a['all_unambiguous_notes'].items()})
        print(key, 'cores', {k: round(v, 4) for k, v in a['all_note_cores'].items()})
        print(key, 'ideal follower', {k: round(v, 4) for k, v in a['ideal_follower_of_her_pitch'].items()})
        print(key, 'female', {k: round(v, 4) for k, v in a['female_only'].items()})
        print(key, 'male', {k: round(v, 4) for k, v in a['male_only'].items()})
    worst = sorted(res['audio']['per_excerpt'], key=lambda r: r['good'])[:5]
    for r in worst:
        print('worst', r['id'], r['mode'], r['avg_pitch'], round(r['good'], 3), round(r['harsh'], 3), round(r['silent'], 3))


def audio_mode(mode):
    return audio(mode=mode)


if __name__ == '__main__':
    main()
