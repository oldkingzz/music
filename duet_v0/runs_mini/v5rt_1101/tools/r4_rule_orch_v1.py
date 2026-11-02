"""Orchestrator's own check of the round-4 selection, read straight from the files on disk (only keep_table_v1, which
the orchestrator wrote in round 2, is reused for G1).

Rule (state_R3.json, round4_rule_fixed_now); val only, tuned on val, no test split. W = r3_X_regtrkst_v1.
  K1: mean rt_nd_live over {micbt_cal, bt_cal} x {sung, sung_low} >= W's + 0.005.
  K2: in each of those cells harsh <= W + 0.003, fresh_per100 <= W + 0.5, stumbles_per100 <= W + 1.0.
  K3: v11 fresh_heard <= 4, harsh_heard <= 0.1602.
  K4: bench_rt L 0.5 vs reactive:r2_mix_v2 at L 0.5: rt_nd >= +0.005, harsh <=, fresh clashes <=, seed2 delta >= +0.003.
  Rank: highest K1; within 0.003 of the top, fewer fresh + stumbles summed over the 4 cells.
  Use: pick passes round-3 G1 / G2 / G3 at wired and wired mean >= W - 0.003 -> single default; else above 120 ms.
Also prints the verifier's tempo probe (pick vs W) side by side.

  .venv/bin/python runs_mini/v5rt_1101/tools/r4_rule_orch_v1.py --out <new json>
"""
import argparse
import json
import os
import sys

ABS = '/Users/san/claude_sandbox/music/duet_v0'
RUN = os.path.join(ABS, 'runs_mini', 'v5rt_1101')
sys.path.insert(0, os.path.join(RUN, 'tools'))
import keep_table_v1 as KT  # noqa: E402

V1 = {'live_bench_rt.py': '46b43f91bf4bfa03f04c07ecdb543368394b6279f198cc8a21c435a8ab2bd8fd',
      'voice_v1.py': 'ed395a5acb7c4a1dbf341c22f41b3b772f74feaeb492880ff93ae8799220e2f9',
      'bench_rt.py': 'f1eb524aa939b5ba54704532ac7b5eb536659ceaaef32c3701934f9cf7d3e772'}
V2 = 'ffeb1ab0b6585719afcb6baf974cf1fc0a373852822eb12b32875572ed442946'
W = 'r3_X_regtrkst_v1'
OLD = ('r3_X_regtrkst_v1', 'r3_X_regtrkdk_v1', 'r3_X_regtrkstsafe_v1')
NEW = {'G': ('hedge', 'ehedge', 'ehedgesafe'), 'H': ('lagk15', 'lagk15hd', 'lagk15hq', 'lagk2hd'),
       'X': ('hgeh', 'hgsafe', 'hdgsafe')}
CELLS = [(v, lat) for lat in ('micbt_cal', 'bt_cal') for v in ('sung', 'sung_low')]


def load(rel):
    with open(os.path.join(RUN, rel)) as fh:
        return json.load(fh)


def live(rel, spec, voice, lat, v2):
    d = load('live_bench/' + rel)
    assert d['spec'] == spec and d['voice'] == voice and d['lat'] == lat and d['split'] == 'val', rel
    assert all(d['code'][k] == h for k, h in V1.items()), rel
    if v2:
        assert d['code'].get('live_bench_rt_v2.py') == V2, rel
    if voice == 'phone_v11':
        p = d['phone']['proxies']
        return {'fresh_heard': p['fresh_heard'], 'harsh_heard': p['harsh_heard']}
    assert d['songs_n'] == 17 and d['bpm'] == 90.0 and d['count_in'] == 4, rel
    h = d['headline']
    return {k: h[k] for k in ('rt_nd_live', 'harsh', 'fresh_per100', 'stumbles_per100')}


def files(name):
    """(cell prefix, v11 file, K4 prefix, wired prefix) for a pool name."""
    if name in OLD:
        k4 = 'L4_rt_' if name == W else 'R4X_rt_'
        return 'L4_rt_', f'R3X_rt_{name}_phone_v11.json', k4, 'R3X_rt_'
    p = f'R4{name[3]}_rt_'
    return p, f'{p}{name}_phone_v11.json', p, p


def bench(prefix, name, L):
    s, s2, a = (load(f'scores/{prefix}{name}_L{L}.json'), load(f'scores/{prefix}{name}_L{L}_seed2.json'),
                load(f'attack/{prefix}{name}_L{L}.json'))
    assert s['L'] == L and s['split'] == 'val' and s['seed'] == 'seed1' and s2['seed'] == 'seed2' and a['L'] == L
    assert s['bench_sha256'] == V1['bench_rt.py']
    return {'rt_nd': s['headline']['rt_nd'], 'harsh': s['headline']['harsh'], 'seed2': s2['headline']['rt_nd'],
            'fresh': a['headline']['fresh_clash_per100']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out_path = ap.parse_args().out
    react = {f'{v}@{lat}': live(f'L4_reactive_r2_mix_v2_{v}_{lat}.json', 'reactive:r2_mix_v2', v, lat, True)
             for v, lat in CELLS}
    r05 = bench('L4_', 'reactive_r2_mix_v2', 0.5)
    rw = {v: live(f'L_reactive_r2_mix_v2_{v}_wired.json', 'reactive:r2_mix_v2', v, 'wired', False)
          for v in ('sung', 'sung_low')}
    rv11 = live('L_reactive_r2_mix_v2_phone_v11.json', 'reactive:r2_mix_v2', 'phone_v11', 'wired', False)
    b1 = KT.row(KT.REACT)
    pool = list(OLD) + [f'r4_{k}_{v}_v1' for k, vs in NEW.items() for v in vs]
    rows = {}
    for name in pool:
        cp, v11f, k4p, wp = files(name)
        spec = f'rt:{name}'
        cells = {f'{v}@{lat}': live(f'{cp}{name}_{v}_{lat}.json', spec, v, lat, True) for v, lat in CELLS}
        v11 = live(v11f, spec, 'phone_v11', 'wired', False)
        k4 = bench(k4p, name, 0.5)
        wired = {v: live(f'{wp}{name}_{v}_wired.json', spec, v, 'wired', False) for v in ('sung', 'sung_low')}
        g1 = KT.checks(KT.row((f'scores/{wp}{name}_L0.1.json', f'scores/{wp}{name}_L0.1_seed2.json',
                               f'attack/{wp}{name}_L0.1.json')), b1)
        rows[name] = {'cells': cells, 'v11': v11, 'k4': k4, 'wired': wired, 'g1_ok': all(g1.values()),
                      'k1_mean': sum(c['rt_nd_live'] for c in cells.values()) / 4,
                      'k_fs': sum(c['fresh_per100'] + c['stumbles_per100'] for c in cells.values()),
                      'wired_mean': (wired['sung']['rt_nd_live'] + wired['sung_low']['rt_nd_live']) / 2}
    w = rows[W]
    for name, r in rows.items():
        fail = []
        if r['k1_mean'] < w['k1_mean'] + 0.005:
            fail.append('K1')
        for c, x in r['cells'].items():
            y = w['cells'][c]
            if x['harsh'] > y['harsh'] + 0.003 or x['fresh_per100'] > y['fresh_per100'] + 0.5 \
                    or x['stumbles_per100'] > y['stumbles_per100'] + 1.0:
                fail.append(f'K2 {c}')
        if r['v11']['fresh_heard'] > 4 or r['v11']['harsh_heard'] > 0.1602:
            fail.append('K3')
        k = r['k4']
        if not (k['rt_nd'] >= r05['rt_nd'] + 0.005 and k['harsh'] <= r05['harsh'] and k['fresh'] <= r05['fresh']
                and k['seed2'] - r05['seed2'] >= 0.003):
            fail.append('K4')
        ws, wl = r['wired']['sung'], r['wired']['sung_low']
        g2 = (ws['rt_nd_live'] >= rw['sung']['rt_nd_live'] + 0.005 and wl['rt_nd_live'] >= rw['sung_low']['rt_nd_live']
              and all(r['wired'][v]['harsh'] <= rw[v]['harsh'] and r['wired'][v]['fresh_per100'] <=
                      rw[v]['fresh_per100'] for v in ('sung', 'sung_low')))
        g3 = r['v11']['fresh_heard'] <= rv11['fresh_heard'] and r['v11']['harsh_heard'] <= rv11['harsh_heard'] + 0.005
        r.update({'failed': fail, 'passes_K': not fail, 'wired_G': r['g1_ok'] and g2 and g3})
    ok = {n: r for n, r in rows.items() if r['passes_K'] and n != W}
    top = max(r['k1_mean'] for r in ok.values()) if ok else None
    band = sorted((n for n, r in ok.items() if top - r['k1_mean'] <= 0.003), key=lambda n: rows[n]['k_fs']) if ok else []
    pick = band[0] if band else None
    use = 'none'
    if pick:
        use = 'single default' if rows[pick]['wired_G'] and rows[pick]['wired_mean'] >= w['wired_mean'] - 0.003 \
            else 'above 120 ms'
    print(f'W K1 {w["k1_mean"]:.4f} (bar {w["k1_mean"] + 0.005:.4f}); reactive K1 '
          f'{sum(c["rt_nd_live"] for c in react.values()) / 4:.4f}; reactive L0.5 {r05}')
    print(f"{'system':22s} K1      fs     K4rt   wired  wG  failed")
    for n in sorted(rows, key=lambda n: -rows[n]['k1_mean']):
        r = rows[n]
        print(f"{n:22s} {r['k1_mean']:.4f} {r['k_fs']:6.2f} {r['k4']['rt_nd']:.4f} {r['wired_mean']:.4f} "
              f"{'Y' if r['wired_G'] else '.'}   {', '.join(r['failed']) or '-'}")
    print(f'pass K1-K4 {len(ok)}, top {top}, band {band}, PICK {pick}, use {use}')
    for c in react:
        x, y, z = rows[pick]['cells'][c] if pick else None, w['cells'][c], react[c]
        print(f'{c:18s} pick {x} | W {y} | reactive {z}')
    tp = {}
    for who, f in (('pick', 'VER4_tempo_probe_hgeh_v1.json'), ('W', 'VER4_tempo_probe_W_v2.json')):
        d = load('verify/' + f)
        tp[who] = {c: {k: v for k, v in x.items() if k != 'per_song'} for c, x in d['combos'].items()}
    print('tempo probe (verifier, report only): combo: pick rt_nd/harsh/fresh/stumbles vs W')
    for c in sorted(set(tp['pick']) & set(tp['W'])):
        a, b = tp['pick'][c], tp['W'][c]
        print(f"  {c:24s} {a['rt_nd_live']:.4f}/{a['harsh']:.4f}/{a['fresh_per100']:.2f}/{a['stumbles_per100']:.2f}  vs  "
              f"{b['rt_nd_live']:.4f}/{b['harsh']:.4f}/{b['fresh_per100']:.2f}/{b['stumbles_per100']:.2f}")
    out = {'rows': rows, 'reactive_cells': react, 'reactive_L05': r05, 'top_k1': top, 'band': band, 'pick': pick,
           'use': use, 'tempo_probe': tp,
           'note': 'val only (17 Bach chorales at 90 bpm; bench_rt val sets); tuned on val; no test split'}
    with open(out_path, 'x') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
