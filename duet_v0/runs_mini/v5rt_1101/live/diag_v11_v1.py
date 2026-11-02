"""diag_v11_v1 (run v5rt_1101, live phase): where the live engines' differences on the v11 phone file come from.

futures/v4_selfiter/live_rt_eval_v2.py found that on v11 (a real singer, NO ground truth) RTDuet(r2_copyhmbold_v1) has
more fresh clashes and more harsh time against the notes the tracker found than reactive r2_mix_v2, while it is far
better on the Twinkle phone file. This runs the same engines on the same files with live_rt_eval_v2's code (imported
unchanged; it installs the no-sound tripwire; NO sound device is opened) and
  - attributes every Duet note to the decision that started it (RTDuet: its decision log; PredictDuet: its beat plans;
    reactive: every note is a hear),
  - splits the fresh clashes against her heard pitch, and the harsh time (exact overlap of Duet's notes with the notes
    the segmenter closed, against their heard pitch), by that origin,
  - counts which internal path the champion's hear() took (its own stats counter, before/after each run),
  - lists v11 note by note: what the tracker heard and what reactive and the champion sang.
Proxies only on v11. Twinkle is run too, for the same split where the truth is known (its truth metrics are in
live_rt_eval_v2's output).

  cd /Users/san/claude_sandbox/music/duet_v0 && .venv/bin/python runs_mini/v5rt_1101/live/diag_v11_v1.py --out <new json>
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ABS = '/Users/san/claude_sandbox/music/duet_v0'
sys.path.insert(0, os.path.join(ABS, 'futures', 'v4_selfiter'))
import live_rt_eval_v2 as E  # noqa: E402  (installs the no-sound tripwire before anything else)
import numpy as np  # noqa: E402

SR, HOP, LO = E.SR, E.HOP, E.LO


def origins(kind, eng, notes):
    """The decision that started each Duet note (start s, end s, midi)."""
    if kind == 'rt':
        by_t = defaultdict(list)
        for d in eng.decisions:
            by_t[d['t']].append(d)
        out = []
        for a, e, m in notes:
            ds = [d for d in by_t.get(round(a, 4), []) if d['duet'] == m]
            d = ds[-1] if ds else None
            out.append('?' if d is None else d['kind'] + '/' + d['action'])
        return out
    if kind == 'predict':
        starts = set()
        for p in eng.plans:
            if p.get('h') is not None and p['h'] >= 0:
                i0 = max(int(round((eng.t0 + p['k'] * eng.T_b - eng.Lo - eng.PRE) * SR)), 0)
                starts.add(round(i0 / SR + eng.Lo, 4))
        return ['plan' if round(a, 4) in starts else 'hear' for a, e, m in notes]
    return ['hear'] * len(notes)


def analyse(kind, label, sysm, name, song, x, beat1=None):
    st0 = Counter(getattr(sysm, 'stats', {}))
    eng = E.make(kind, sysm, name, song)
    track, log, _, _ = E.run(eng, x)
    st1 = Counter(getattr(sysm, 'stats', {}))
    notes, blips = E.voice_notes(log, LO)
    org = origins(kind, eng, notes)
    her = [(float(on), float(off), int(m), None) for on, off, m in eng.notes]
    att = []
    for j, (a, e, m) in enumerate(notes):
        prev = None
        for p in notes[:j]:
            if p[0] < a <= p[1]:
                prev = p
        if prev is None or prev[2] != m:
            att.append(j)
    clashes = []
    att_by = Counter()
    for j in att:
        a, e, m = notes[j]
        nt = E.her_at(a, her)
        if nt is None:
            continue
        att_by[org[j]] += 1
        iv = abs(nt[2] - m) % 12
        if iv in E.HARSH:
            clashes.append({'t': round(a, 3), 'beat': round((a - eng.t0) / eng.T_b, 2), 'her': E.nn(nt[2]),
                            'her_on_beat': round((nt[0] - eng.t0) / eng.T_b, 2), 'duet': E.nn(m), 'iv': iv,
                            'origin': org[j]})
    sing = sum(off - on for on, off, _, _ in her)
    harsh_by, sound_by = defaultdict(float), defaultdict(float)
    for (a, e, m), o in zip(notes, org):
        for on, off, hm, _ in her:
            ov = min(e, off) - max(a, on)
            if ov > 0:
                sound_by[o] += ov
                if abs(hm - m) % 12 in E.HARSH:
                    harsh_by[o] += ov
    t_out = (np.arange(len(track)) + 2) * HOP / SR + LO
    rows = []
    for on, off, hm, _ in her:
        ev = next((e for e in eng.events if abs(eng.t0 + e['onset_raw'] * eng.T_b - on) < 0.006), None)
        v0, v1 = E.at_time(track, t_out, on), E.at_time(track, t_out, min(on + E.W_AFTER, off))
        rows.append({'on': round(on, 3), 'beat': round((on - eng.t0) / eng.T_b, 2), 'dur_ms': round((off - on) * 1000),
                     'her': E.nn(hm), 'heard_at_answer': E.nn(ev['her_heard']) if ev else None,
                     'action': (ev.get('action') if ev else None), 'at_onset': E.nn(v0), 'after_200ms': E.nn(v1),
                     'iv_after': None if v1 < 0 else abs(hm - v1) % 12})
    return {'engine': label, 't0': eng.t0, 'her_notes': len(her), 'her_singing_s': round(sing, 3),
            'duet_notes': len(notes), 'attacks_on_her': dict(att_by), 'fresh_clashes': clashes,
            'harsh_s_by_origin': {k: round(v, 3) for k, v in harsh_by.items()},
            'sound_s_by_origin': {k: round(v, 3) for k, v in sound_by.items()},
            'harsh_share_exact': round(sum(harsh_by.values()) / sing, 4) if sing else None,
            'system_paths': {k: st1[k] - st0[k] for k in st1 if st1[k] - st0[k]}, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    path = os.path.abspath(a.out)
    root = os.path.join(ABS, 'runs_mini', 'v5rt_1101', 'live') + os.sep
    if not path.startswith(root) or not path.endswith('.json') or os.path.exists(path):
        sys.exit('--out must be a new .json under ' + root)
    train = E.bench3.train_sets()
    systems = {'r2_mix_v2': E.LB.build('r2_mix_v2', train)}
    for name in ('r0_predict_live_v1', 'r2_copyhmbold_v1'):
        systems[name] = E.LR.rt_build(name, train)
    out = {'note': 'proxies against the notes the tracker found (no ground truth on v11); no sound device used'}
    for key, song in (('v11', E.V11), ('twinkle', E.TW)):
        x = E.read(song['file'])
        out[key] = {}
        for kind, label, name in E.ENGINES:
            r = analyse(kind, label, systems[name], name, song, x)
            out[key][label] = r
            print(f"{key:7s} {label:27s} her notes {r['her_notes']} ({r['her_singing_s']:.1f} s) Duet notes "
                  f"{r['duet_notes']} attacks on her {r['attacks_on_her']} harsh exact {r['harsh_share_exact']} "
                  f"harsh s by origin {r['harsh_s_by_origin']}", flush=True)
            print(f"        sound s by origin {r['sound_s_by_origin']}", flush=True)
            if r['system_paths']:
                print(f"        system paths {r['system_paths']}", flush=True)
            for c in r['fresh_clashes']:
                print(f"        clash beat {c['beat']:6.2f} (her note from beat {c['her_on_beat']:6.2f}) she {c['her']:4s} "
                      f"Duet {c['duet']:4s} iv {c['iv']:2d}  from {c['origin']}", flush=True)
    print('\nv11 note by note (her = segmenter note, heard = pitch when answered; Duet at her onset -> 200 ms later)')
    rr, rc = out['v11']['reactive r2_mix_v2']['rows'], out['v11']['RTDuet r2_copyhmbold_v1']['rows']
    for a_, b_ in zip(rr, rc):
        print(f"  beat {a_['beat']:6.2f} {a_['dur_ms']:4d} ms she {a_['her']:4s} heard {str(a_['heard_at_answer']):5s} | "
              f"reactive {a_['at_onset']:4s}->{a_['after_200ms']:4s} iv {str(a_['iv_after']):4s} | champion "
              f"{b_['at_onset']:4s}->{b_['after_200ms']:4s} iv {str(b_['iv_after']):4s} {str(b_['action'])}", flush=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1, default=E._js)
    print(f'wrote {path}', flush=True)


if __name__ == '__main__':
    main()
