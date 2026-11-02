"""Keep rule (v4_selfiter §4.2, never changed during the run) for one candidate against the champion, from files only:
  nd      runs_mini/v4_1101/nd/<name>/results/nd_<name>.json          (rescore_nd.py, §2.3)
  bench3  last val row of futures/v3_nolatency/leaderboard_v3.jsonl    (big, par, flags)
  seed2   runs_mini/v4_1101/seed2/<name>.json                         (futures/v4_selfiter/seed2.py)
  review  runs_mini/v4_1101/review/<name>.json                        (tools/review.py: (a), (b), (d))
--gate checks only the numeric part that needs nd + bench3 (who goes to review). Prints one JSON verdict.

  cd $ABS && .venv/bin/python runs_mini/v4_1101/tools/keep.py <candidate> <champion> [--gate]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.dirname(HERE)
ABS = os.path.dirname(os.path.dirname(RUNDIR))
BOARD = os.path.join(ABS, 'futures', 'v3_nolatency', 'leaderboard_v3.jsonl')
DS = ('bach', 'folk', 'pop', 'pop909')


def nd(name):
    p = os.path.join(RUNDIR, 'nd', name, 'results', f'nd_{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def row(name):
    last = None
    for line in open(BOARD):
        e = json.loads(line)
        if e['system'] == name and e['split'] == 'val':
            last = e
    return last


def other(sub, name):
    p = os.path.join(RUNDIR, sub, f'{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def verdict(cand, champ, gate=False):
    c, h, cr, hr = nd(cand), nd(champ), row(cand), row(champ)
    missing = [w for w, x in (('nd ' + cand, c), ('nd ' + champ, h), ('bench3 ' + cand, cr), ('bench3 ' + champ, hr))
               if x is None]
    checks = []

    def add(what, val, thr, ok):
        checks.append({'check': what, 'value': round(val, 5), 'threshold': round(thr, 5), 'ok': bool(ok)})

    if not missing:
        add('headline_nd >= champ + 0.005', c['headline_nd'], h['headline_nd'] + 0.005,
            c['headline_nd'] >= h['headline_nd'] + 0.005)
        for ds in ('folk', 'pop'):
            add(f'{ds} >= champ + 0.003', c[ds]['music_nd'], h[ds]['music_nd'] + 0.003,
                c[ds]['music_nd'] >= h[ds]['music_nd'] + 0.003)
        for ds in DS:
            add(f'{ds} drop <= 0.01', c[ds]['music_nd'], h[ds]['music_nd'] - 0.01,
                c[ds]['music_nd'] >= h[ds]['music_nd'] - 0.01)
            add(f'{ds} double <= 0.08', c[ds]['double'], 0.08, c[ds]['double'] <= 0.08)
            for k in ('big', 'par'):
                add(f'{ds} {k} <= champ + 0.01', cr['rows'][ds][k], hr['rows'][ds][k] + 0.01,
                    cr['rows'][ds][k] <= hr['rows'][ds][k] + 0.01)
        add('no SLOW flag', max(r['us_per_note'] for r in cr['rows'].values()), 2000.0, 'SLOW' not in cr['flags'])
    if not gate:
        s, sh, rv = other('seed2', cand), other('seed2', champ), other('review', cand)
        missing += [w for w, x in (('seed2 ' + cand, s), ('seed2 ' + champ, sh), ('review ' + cand, rv)) if x is None]
        if s and sh:
            add('seed2 cand - champ >= 0.003', s['headline_nd'] - sh['headline_nd'], 0.003,
                s['headline_nd'] - sh['headline_nd'] >= 0.003)
        if rv:
            for k in ('a_causal', 'b_grep', 'd_range'):
                add(f'review {k}', float(rv[k]['ok']), 1.0, rv[k]['ok'])
    ok = not missing and all(x['ok'] for x in checks)
    return {'candidate': cand, 'champion': champ, 'gate_only': gate, 'missing': missing,
            'failed': [x['check'] for x in checks if not x['ok']], 'pass': ok, 'checks': checks,
            'headline_nd': c and round(c['headline_nd'], 4), 'champ_headline_nd': h and round(h['headline_nd'], 4)}


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    print(json.dumps(verdict(a[0], a[1], '--gate' in sys.argv), indent=1))
