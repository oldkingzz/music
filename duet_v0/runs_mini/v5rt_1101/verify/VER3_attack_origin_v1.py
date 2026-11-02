"""VER3_attack_origin_v1 (verifier, run v5rt_1101, round 3): where do the live fresh clashes and stumbles of the winner
rt:r3_X_regtrkst_v1 and of reactive:r2_mix_v2 come from? Reads the VER3_* live_bench files (voices clean, sung,
sung_low; wired) and re-applies live_bench_rt.score()'s attack rules (live_rt_eval_v2.attacks / her_at, HARSH, the
stumble window) to Duet's notes, splitting every attack by origin: 'plan' = a note RTDuet started at a plan point (it
sounds PRE = 25 ms before the half beat q, i.e. at beat q - 0.0375 at 90 bpm), 'other' = a hear (or re-entry) attack,
which starts on the engine's 10 ms output-block grid. The totals must equal the bench's own counts. Val only (17 val Bach
chorales); systems tuned on val; no test split. Read-only on its inputs; writes ONE new JSON (mode 'x').

  .venv/bin/python runs_mini/v5rt_1101/verify/VER3_attack_origin_v1.py --out runs_mini/v5rt_1101/verify/<new>.json
"""
import sys

sys.path.insert(0, '/Users/san/claude_sandbox/music/duet_v0/futures/v5_rt')
import live_bench_rt as LBR  # noqa: E402  (tripwire first)
import argparse  # noqa: E402
import bisect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter  # noqa: E402

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
OUT_DIR = os.path.join(RUN, 'verify')
PRE = 0.025


def analyse(fname, voice):
    with open(os.path.join(RUN, fname)) as fh:
        res = json.load(fh)
    T_b = 60.0 / LBR.BPM
    t0h = 0.5 + LBR.COUNT_IN * T_b
    shift = -12 if voice == 'sung_low' else 0
    songs = {s['id']: s for s in LBR.bench3.eval_sets('val')['bach']}
    c = Counter()
    for r in res['songs']:
        song = songs[r['id']]
        her = LBR.VV.true_notes(song, t0h, T_b, shift)
        z = [b['z'] for b in song['beats']]
        ton = song['tonic']
        duet = sorted((t0h + a * T_b, t0h + e * T_b, int(m)) for a, e, m in r['duet'])
        Ts, H = LBR.step_fn(duet)
        b = lambda x: (x - t0h) / T_b  # noqa: E731
        hv = [(on_s, off_s, m, non) for on_s, off_s, m, non, _ in her]
        ons = [n[0] for n in her]
        counted = [non >= LBR.J and int(math.floor(non + 1e-9)) < len(z) and z[int(math.floor(non + 1e-9))] >= 0
                   for _, _, _, non, _ in her]
        for x, h in LBR.EV.attacks(duet):
            g = b(x) + PRE / T_b
            kind = 'plan' if abs(2 * g - round(2 * g)) < 0.002 else 'other'
            c[f'attacks_{kind}'] += 1
            nt = LBR.EV.her_at(x, hv)
            if nt is not None:
                xb = max(b(x), nt[3])
                kb = int(math.floor(xb))
                if not (xb < LBR.J or kb >= len(z) or z[kb] < 0):
                    c[f'att_{kind}'] += 1
                    if abs(nt[2] - h) % 12 in LBR.HARSH:
                        c[f'fresh_{kind}'] += 1
                    c[f'wrong_{kind}'] += not LBR.BR.hit(h, nt[2], ton, z[kb])
                    c[f'above_{kind}'] += h > nt[2]
            j = bisect.bisect_right(ons, x + LBR.ST_PRE) - 1
            if j >= 0 and counted[j] and x <= ons[j] + LBR.ST_POST:
                q = bisect.bisect_right(Ts, x) - 1
                while q + 1 < len(Ts) and H[q + 1] == h:
                    q += 1
                t_change = Ts[q + 1] if q + 1 < len(Ts) else math.inf
                if t_change - x < LBR.ST_WIN:
                    c[f'stumbles_{kind}'] += 1
    m = res['headline']
    out = dict(sorted(c.items()))
    out['check_fresh_total'] = [c['fresh_plan'] + c['fresh_other'], m['fresh']]
    out['check_stumbles_total'] = [c['stumbles_plan'] + c['stumbles_other'], round(m['stumbles_per100'] * m['notes'] / 100)]
    out['check_wrong_total'] = [c['wrong_plan'] + c['wrong_other'], round(m['wrong_att_per100'] * m['notes'] / 100)]
    out['notes'] = m['notes']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    path = os.path.abspath(ap.parse_args().out)
    if not path.startswith(OUT_DIR + os.sep) or not os.path.basename(path).startswith('VER3_') \
            or not path.endswith('.json') or os.path.exists(path):
        sys.exit(f'--out must be a new VER3_*.json under {OUT_DIR}')
    out = {'note': 'val only (17 val Bach chorales); systems tuned on val; no test split'}
    for who in ('rt_r3_X_regtrkst_v1', 'reactive_r2_mix_v2'):
        for v in ('clean', 'sung', 'sung_low'):
            key = f'{who}|{v}'
            out[key] = analyse(f'live_bench/VER3_{who}_{v}_wired.json', v)
            print(key, json.dumps(out[key]), flush=True)
    with open(path, 'x') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
