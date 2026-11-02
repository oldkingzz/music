"""The audio path, end to end: synthetic singer -> YIN front end -> chord HMM + planner -> Duet's voice.

1. Front-end check on all test chorales, for a clean voice (A4 holds) and for the repo's singer with
   vibrato and glides (A4 broken): does the beat observation from audio match the score, and how
   much do the scores move when Duet listens to audio instead of reading the score?
2. The 'slow harmony' style setting scored on the Bach test set (it should hurt there).
3. Demos with audio: Twinkle, your arpeggios, and one typical Bach chorale (picked by rule).
Writes duet_beat/out/audio_results.json and duet_beat/out/media/*.mp3 (wav copies are git-ignored).
"""
import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audio import AudioSource, click, duet_notes, mix, render_her  # noqa: E402
from data import load  # noqa: E402
from metrics import bootstrap, song_counts, summarize  # noqa: E402
from model import ChordHMM, soprano_obs  # noqa: E402
from planner import Config, run  # noqa: E402
from songs import arpeggios, slow_harmony, twinkle  # noqa: E402
from theory import CHORDS  # noqa: E402

OUT = os.path.join(HERE, 'out')
MEDIA = os.path.join(OUT, 'media')


def cfg_from(d):
    d = dict(d)
    d['alto'] = tuple(d['alto'])
    return Config(**d)


def place(song, bpm):
    T_b = 60.0 / bpm
    count_in = song['M']
    return T_b, count_in * T_b, count_in


def front_end_check(test, model, cfg, style):
    agree = {'full': [0, 0], 'partial': [0, 0]}
    alarms, drifts = 0, []
    mism = [0, 0]                                   # full-beat mismatches, and how many sit in beats with 2+ notes
    cs_audio, cs_score = [], []
    T_b, t0, count_in = place(test[0], cfg.bpm)
    D = (cfg.L + cfg.delta) / T_b
    for song in test:
        T_b, t0, _ = place(song, cfg.bpm)
        her = render_her(song, t0, T_b, style)
        src = AudioSource(her, t0, T_b, song['K'])
        for k in range(song['K']):
            for kind, frac in (('full', 1.0), ('partial', 1.0 - D)):
                _, ms = soprano_obs(song['sop'], k, frac)
                _, ma = src.full(k) if kind == 'full' else src.partial(k, frac)
                if ms is not None:
                    agree[kind][1] += 1
                    hit = ma is not None and (ma - ms) % 12 == 0
                    agree[kind][0] += int(hit)
                    if kind == 'full' and not hit:
                        mism[0] += 1
                        mism[1] += sum(1 for on, off, m in song['sop'] if on < k + 1 - 1e-9 and off > k + 1e-9) >= 2
        beats = run(song, model, cfg, 'hmm', source=src)
        alarms += any(b.action == 'stopped' for b in beats)
        drifts += [abs(b.drift) for b in beats if b.drift == b.drift]
        cs_audio.append(song_counts(song, beats, cfg.J))
        cs_score.append(song_counts(song, run(song, model, cfg, 'hmm'), cfg.J))
    return {'agree_full': agree['full'][0] / agree['full'][1], 'agree_partial': agree['partial'][0] / agree['partial'][1],
            'n_beats': agree['full'][1], 'audio': {**summarize(cs_audio), 'ci': bootstrap(cs_audio)},
            'mismatch': mism[0], 'mismatch_multi': mism[1], 'false_alarms': alarms, 'n_songs': len(test), 'drift_median_abs_ms': float(np.median(drifts)) if drifts else None,
            'drift_p95_abs_ms': float(np.percentile(drifts, 95)) if drifts else None,
            'score': summarize(cs_score)}


def demo(song, model, cfg, name, style='clean', title=None, her_bpm=None, grid=None):
    bpm = song.get('bpm', cfg.bpm)
    cfg = replace(cfg, bpm=bpm)
    T_b, t0, count_in = place(song, bpm)
    her = render_her(song, t0, 60.0 / her_bpm if her_bpm else T_b, style)
    src = AudioSource(her, t0, T_b, song['K'])
    src.sync_grid = grid or song.get('grid', 0.5)
    beats = run(song, model, cfg, 'hmm', source=src)
    sm = summarize([song_counts(song, beats, cfg.J)])
    n = len(her) + int(0.5 * 24000)
    clk = click(n, t0, T_b, song['K'], song['M'], count_in)
    os.makedirs(MEDIA, exist_ok=True)
    wav, mp3 = os.path.join(MEDIA, f'{name}.wav'), os.path.join(MEDIA, f'{name}.mp3')
    mix(her, duet_notes(beats, t0, T_b), clk, wav, mp3)
    return {
        'name': name, 'title': title or song.get('title') or song['id'], 'id': song['id'], 'style': style,
        'tonic': song['tonic'], 'M': song['M'], 'K': song['K'], 'bpm': bpm, 't0': t0, 'T_b': T_b, 'J': cfg.J,
        'model': getattr(model, 'style', 'bach'), 'audio': f'media/{name}.mp3', 'metrics': sm, 'her_bpm': her_bpm or bpm,
        'sop': song['sop'],
        'beats': [{'k': b.k, 'pos': b.pos, 'z_true': b.z_true, 'z_hat': b.z_hat, 'q': round(b.q, 3),
                   'action': b.action, 'h': b.h, 'pi': b.pi, 'drift': b.drift} for b in beats],
    }


def main():
    d = load()['split']
    model = ChordHMM().fit(d['train'])
    res = json.load(open(os.path.join(OUT, 'results.json')))
    cfg = cfg_from(res['tuning']['chosen']['main'])
    slow = slow_harmony(model)
    out = {'cfg': asdict(cfg)}

    # 1. front end
    out['front_end'] = {}
    for style in ('clean', 'vibrato'):
        fe = front_end_check(d['test'], model, cfg, style)
        out['front_end'][style] = fe
        print(style, 'agree full %.3f partial %.3f' % (fe['agree_full'], fe['agree_partial']),
              'audio M1 %.3f M2 %.3f U %.3f' % (fe['audio']['m1'], fe['audio']['m2'], fe['audio']['u']),
              'score M1 %.3f M2 %.3f U %.3f' % (fe['score']['m1'], fe['score']['m2'], fe['score']['u']), flush=True)

    # 2. the slow-harmony style on Bach (score input)
    cs = [song_counts(s, run(s, slow, cfg, 'hmm'), cfg.J) for s in d['test']]
    out['slow_on_bach'] = {**summarize(cs), 'ci': bootstrap(cs)}
    print('slow model on Bach test: M1 %.3f M2 %.3f U %.3f' % (out['slow_on_bach']['m1'], out['slow_on_bach']['m2'], out['slow_on_bach']['u']))

    # 3. demos
    per = res['per_song_main']
    us = {k: v['u'] for k, v in per.items()}
    med = float(np.median(list(us.values())))
    typical = min(d['test'], key=lambda s: (abs(us[s['id']] - med), s['id']))
    demos = []
    tw, ar = twinkle(), arpeggios()
    for song, mdl, name in ((tw, model, 'twinkle_bach'), (tw, slow, 'twinkle_slow'),
                            (ar, model, 'arp_bach'), (ar, slow, 'arp_slow')):
        for mode in ('long', 'vec'):
            c = replace(cfg, mode=mode, lam=res['tuning']['chosen']['vec']['lam'] if mode == 'vec' else cfg.lam)
            dm = demo(song, mdl, c, f'{name}_{mode}')
            dm['mode'] = mode
            demos.append(dm)
            print(dm['name'], {k: round(v, 3) for k, v in dm['metrics'].items() if k in ('m1', 'm2', 'm5', 'u')}, flush=True)
    dm = demo(typical, model, cfg, 'bach_typical', title=f"Bach chorale {typical['id'].split('/')[-1].upper()} (test set, typical by rule)")
    dm['mode'] = 'long'
    dm['rule'] = f'test chorale whose U is closest to the median U ({med:.3f}) of the main system on the score'
    demos.append(dm)
    print(dm['name'], typical['id'], {k: round(v, 3) for k, v in dm['metrics'].items() if k in ('m1', 'm2', 'm5', 'u')})
    dm = demo(tw, slow, cfg, 'twinkle_slow_vibrato', style='vibrato')
    dm['mode'] = 'long'
    demos.append(dm)
    print(dm['name'], {k: round(v, 3) for k, v in dm['metrics'].items() if k in ('m1', 'm2', 'm5', 'u')})
    dm = demo(tw, slow, cfg, 'twinkle_drift', her_bpm=101.5)
    dm['mode'] = 'long'
    demos.append(dm)
    first = next((b['k'] for b in dm['beats'] if b['action'] == 'stopped'), None)
    print(dm['name'], 'she sings 101.5 BPM on a 100 BPM click; Duet stops at beat', first)
    out['demos'] = demos
    out['chords'] = list(CHORDS)
    with open(os.path.join(OUT, 'audio_results.json'), 'w') as fh:
        json.dump(out, fh, default=float)


if __name__ == '__main__':
    main()
