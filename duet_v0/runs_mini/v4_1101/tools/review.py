"""Reviewer checks (v4_selfiter §4.1) (a), (b), (d) for one system; (c) is futures/v4_selfiter/seed2.py.

(a) causality: 24 val songs (6 per dataset, fixed seed) x 3 cut notes. Every pitch after the cut note, the duration
    of the cut note and every later timing, and all chord labels are randomized (tonic, meter, K and the bar
    positions are the song's meta and stay). Duet's notes 0..cut must be identical; also checks determinism.
(b) cheat grep over the system file and every systems/ module it imports, recursively. A banned token anywhere
    fails, except in legacy files (unchanged since Phase A, sha256 in tools/legacy_systems.sha256) where only the
    env-knob reads (os.environ: M1_*, M3_*, DUET_HIST, DUET_M4; inert because the environment is checked empty)
    and torch.load of the GRU checkpoints in systems/ are allowed. New files may torch.load only a '.pt' in systems/.
(d) range: over all val songs, every Duet note inside theory.ALTO, and above her HEARD note on <= 5 % of sung notes.
Writes runs_mini/v4_1101/review/<name>.json (refuses if it exists).

  cd $ABS/runs_mini/v4_1101/review && $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/hbrun.py <label> 600 \
      $ABS/.venv/bin/python $ABS/runs_mini/v4_1101/tools/review.py <name>
"""
import copy
import hashlib
import importlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.dirname(HERE)
ABS = os.path.dirname(os.path.dirname(RUNDIR))
V3 = os.path.join(ABS, 'futures', 'v3_nolatency')
SYSDIR = os.path.join(V3, 'systems')
sys.path.insert(0, V3)
import bench3  # noqa: E402
from theory import ALTO  # noqa: E402

BANNED = [r'eval_sets', r'load\(\)', r'_extra', r'bench/data', r'bach_beats', r"'val'", r'"val"', r"'test'",
          r'"test"', r'split', r'os\.environ', r'getenv', r'open\(', r'json\.load', r'torch\.load', r'np\.load',
          r'numpy\.load', r'pickle', r'importlib', r'__import__', r'read_text', r'read_bytes', r'inspect',
          r'_getframe', r'f_locals', r'f_back', r'globals\(\)', r'\bgc\.', r'traceback']
ENV_KNOB = re.compile(r'^(M1_|M3_|DUET_HIST|DUET_M4)')


def legacy():
    out = {}
    for line in open(os.path.join(HERE, 'legacy_systems.sha256')):
        h, f = line.split()
        out[f] = h
    return out


def imports(path):
    src = open(path).read()
    mods = set(re.findall(r'^\s*from\s+systems\.(\w+)\s+import', src, re.M))
    mods |= set(re.findall(r'^\s*import\s+systems\.(\w+)', src, re.M))
    grps = re.findall(r'^\s*from\s+systems\s+import\s+\(([^)]*)\)', src, re.M)
    grps += re.findall(r'^\s*from\s+systems\s+import\s+([^\n(]+)$', src, re.M)
    for grp in grps:
        for item in '\n'.join(x.split('#')[0] for x in grp.splitlines()).replace('\n', ',').split(','):
            if item.strip():
                mods.add(item.strip().split(' as ')[0].strip())
    return mods


def grep(name):
    leg = legacy()
    todo, seen, hits, fails = [name], [], [], []
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.append(mod)
        path = os.path.join(SYSDIR, f'{mod}.py')
        if not os.path.exists(path):
            fails.append({'file': f'{mod}.py', 'why': 'imported module not found in systems/'})
            continue
        raw = open(path, 'rb').read()
        is_legacy = leg.get(f'{mod}.py') == hashlib.sha256(raw).hexdigest()
        for no, text in enumerate(raw.decode().splitlines(), 1):
            for pat in BANNED:
                if re.search(pat, text):
                    h = {'file': f'{mod}.py', 'line': no, 'token': pat, 'text': text.strip()[:160], 'legacy': is_legacy}
                    hits.append(h)
                    if is_legacy:
                        ok = pat == r'os\.environ' or (pat == r'torch\.load' and 'm3_seq_' in raw.decode()
                                                       and mod in ('m3_seq_v1', 'm3_seq_v4'))
                    else:
                        ok = pat == r'torch\.load' and '.pt' in text and ('HERE' in text or 'systems' in text)
                    if not ok:
                        fails.append(h)
        todo += sorted(imports(path) - set(seen))
    env = sorted(k for k in os.environ if ENV_KNOB.match(k))
    if env:
        fails.append({'file': '(environment)', 'why': f'env knobs set: {env}'})
    return {'ok': not fails, 'files': seen, 'hits': hits, 'fails': fails}


def probe_songs():
    sets = bench3.eval_sets('val')
    rng = random.Random(51601)
    out = []
    for ds in ('bach', 'folk', 'pop', 'pop909'):
        for i in sorted(rng.sample(range(len(sets[ds])), 6)):
            out.append((ds, sets[ds][i]))
    return out


def perturb(s0, c, rng):
    """Copy of s0 (sop sorted by onset) with everything after note c's onset randomized, and all chord labels."""
    s = copy.deepcopy(s0)
    sop = s['sop']
    n = len(sop)
    on_c = sop[c][0]
    end = max(x[1] for x in sop)
    if n - c - 1 > 0:
        ons = sorted(rng.uniform(on_c + 0.01, max(on_c + 0.02, end - 0.05)) for _ in range(n - c - 1))
        for j in range(1, len(ons)):
            ons[j] = max(ons[j], ons[j - 1] + 1e-3)
        for j, on in zip(range(c + 1, n), ons):
            sop[j][0] = on
            sop[j][2] = rng.randint(48, 84)
    for j in range(c, n):
        nxt = sop[j + 1][0] if j + 1 < n else min(s['K'], sop[j][0] + 2.0)
        sop[j][1] = sop[j][0] + max(1e-3, rng.uniform(0.15, 1.0) * (nxt - sop[j][0]))
    for b in s['beats']:
        b['z'] = rng.randint(-1, 6)
        b['fit'] = rng.random()
        b['onset_pcs'] = sorted(rng.sample(range(12), rng.randint(0, 4)))
    return s


def causal(system):
    rng = random.Random(5162)
    checks, bad, nondet = 0, [], []
    for ds, song in probe_songs():
        s0 = dict(song, sop=sorted(([*x] for x in song['sop']), key=lambda x: x[0]))
        n = len(s0['sop'])
        if n < 8:
            continue
        hs0 = bench3.run_song(system, s0)
        if bench3.run_song(system, s0) != hs0:
            nondet.append(song.get('id'))
        lo, hi = max(4, n // 5), max(5, n - 2)
        for c in sorted(rng.sample(range(lo, hi), min(3, hi - lo))):
            hs1 = bench3.run_song(system, perturb(s0, c, rng))
            checks += 1
            if hs1[:c + 1] != hs0[:c + 1]:
                first = next(j for j in range(c + 1) if hs1[j] != hs0[j])
                bad.append({'ds': ds, 'song': song.get('id'), 'cut': c, 'n': n, 'first_diff': first})
    return {'ok': not bad and not nondet, 'checks': checks, 'mismatches': bad, 'nondeterministic': nondet}


def rng_check(system):
    sung = above = out = silent = 0
    per = {}
    for ds, songs in bench3.eval_sets('val').items():
        a = [0, 0, 0]
        for song in songs:
            heard = sorted(bench3.B2.heard(song)['sop'], key=lambda x: x[0])
            hs = bench3.run_song(system, song)
            for (on, off, m), h in zip(heard, hs):
                if h < 0:
                    silent += 1
                    continue
                sung += 1
                a[0] += 1
                if not ALTO[0] <= h <= ALTO[1]:
                    out += 1
                    a[1] += 1
                if h > m:
                    above += 1
                    a[2] += 1
        per[ds] = {'sung': a[0], 'out_of_alto': a[1], 'above_share': a[2] / max(a[0], 1)}
    share = above / max(sung, 1)
    return {'ok': out == 0 and share <= 0.05, 'sung': sung, 'silent': silent, 'out_of_alto': out,
            'above_share': share, 'per_dataset': per}


if __name__ == '__main__':
    name = sys.argv[1]
    path = os.path.join(RUNDIR, 'review', f'{name}.json')
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    res = {'name': name, 'b_grep': grep(name)}
    system = importlib.import_module(f'systems.{name}').build(bench3.train_sets())
    res['a_causal'] = causal(system)
    res['d_range'] = rng_check(system)
    res['ok_abd'] = res['a_causal']['ok'] and res['b_grep']['ok'] and res['d_range']['ok']
    with open(path, 'x') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({'name': name, 'ok_abd': res['ok_abd'], 'a_ok': res['a_causal']['ok'],
                      'a_checks': res['a_causal']['checks'], 'b_ok': res['b_grep']['ok'],
                      'b_fails': res['b_grep']['fails'][:5], 'd_ok': res['d_range']['ok'],
                      'out_of_alto': res['d_range']['out_of_alto'],
                      'above_share': round(res['d_range']['above_share'], 4)}))
