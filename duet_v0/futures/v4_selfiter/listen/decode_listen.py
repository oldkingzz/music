"""Decode Vincent's result code from ab.html with key.json (code L1-...) or v2/key.json (code L2-...) and apply the
§3.2 decision rules. Writes listen/result.json (refuses if it exists).

  cd $ABS/futures/v4_selfiter/listen && $ABS/.venv/bin/python $ABS/futures/v3_nolatency/tl.py 60 \
      $ABS/.venv/bin/python $ABS/futures/v4_selfiter/listen/decode_listen.py '<code>'

'Higher-music_nd side' of a pair = the side whose 12 s clip scores higher music_nd; when the two clips are within
0.01 of each other, the side whose system scores higher on the whole val benchmark (EXPECTED.json) instead.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OVERALL = {'m4_oracle': 0.8388, 'm1_filter_v5lo': 0.6206, 'm3_seq_v5': 0.6104, 'third_below': 0.5054}
TAGS = {'c': 'clashing', 'b': 'boring', 'j': 'jumpy', 'r': 'robotic'}

code = sys.argv[1].strip()
mv = re.fullmatch(r'L([12])-([LR]+)(?:-([0-9LRcbjr.]+))?', code)
if not mv:
    sys.exit(f'bad code: {code}')
version, choices, tagstr = int(mv.group(1)), mv.group(2), mv.group(3)
key = json.load(open(os.path.join(HERE, 'key.json' if version == 1 else 'v2/key.json')))
pairs = key['pairs']
if len(choices) != len(pairs):
    sys.exit(f'code has {len(choices)} choices, key has {len(pairs)} pairs')
tags = {}
for tok in (tagstr or '').split('.'):
    if tok:
        t = re.fullmatch(r'(\d+)([LR])([cbjr]+)', tok)
        tags.setdefault(int(t.group(1)), {})[t.group(2)] = [TAGS[x] for x in t.group(3)]

rows, groups = [], {}
for p, ch in zip(pairs, choices):
    win, lose = p[ch], p['R' if ch == 'L' else 'L']
    d = (p['L']['clip_nd'] or 0) - (p['R']['clip_nd'] or 0)
    if abs(d) >= 0.01:
        hi = 'L' if d > 0 else 'R'
        basis = 'clip'
    else:
        dl = OVERALL[p['L']['system']] - OVERALL[p['R']['system']]
        hi = 'L' if dl > 0 else 'R' if dl < 0 else None
        basis = 'overall'
    r = {'pair': p['pair'], 'group': p['group'], 'song': p['song'], 'repeat_of': p['repeat_of'], 'chose': ch,
         'winner': f"{win['system']}:{win['variant']}", 'loser': f"{lose['system']}:{lose['variant']}",
         'clip_nd': {'L': p['L']['clip_nd'], 'R': p['R']['clip_nd']}, 'higher_nd_side': hi, 'basis': basis,
         'chose_higher_nd': hi == ch if hi else None, 'tags': {
             f"{p[s]['system']}:{p[s]['variant']}": v for s, v in tags.get(p['pair'], {}).items()}}
    rows.append(r)
    if p['repeat_of'] is None:
        groups.setdefault(p['group'], []).append(r)

g = {k: {'n': len(v), 'chose_higher_nd': sum(bool(x['chose_higher_nd']) for x in v),
         'wins': {w: sum(x['winner'] == w for x in v) for w in sorted({x['winner'] for x in v} | {x['loser'] for x in v})}}
     for k, v in groups.items()}
rep = next(r for r in rows if r['repeat_of'])
orig = next(r for r in rows if r['pair'] == rep['repeat_of'])
agree = g['m1_vs_third_below']['chose_higher_nd'] >= 3 and g['oracle_vs_m1']['chose_higher_nd'] >= 3
m3_wins = g['m1_vs_m3']['wins'].get('m3_seq_v5:reattack', 0)
tied_wins = g['tied_vs_reattack']['wins'].get('m1_filter_v5lo:tied', 0)
sysagree = {grp: sum(OVERALL[x['winner'].split(':')[0]] > OVERALL[x['loser'].split(':')[0]] for x in groups[grp])
            for grp in ('m1_vs_third_below', 'oracle_vs_m1', 'm1_vs_m3')}
out = {'code': code, 'version': version, 'pairs': rows, 'groups': g, 'chose_higher_system_overall': sysagree,
       'repeat': {'pair': rep['pair'], 'of': orig['pair'], 'same_winner': rep['winner'] == orig['winner']},
       'decision': {'agree': agree, 'low_confidence': rep['winner'] != orig['winner'],
                    'm3_wins_of_4': m3_wins, 'ask_double_ceiling_0.05': m3_wins >= 3,
                    'tied_wins_of_3': tied_wins, 'tied_renders': tied_wins >= 2}}
path = os.path.join(HERE, 'result.json')
if os.path.exists(path):
    sys.exit('refusing: result.json exists')
with open(path, 'x') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({'groups': g, 'repeat': out['repeat'], 'decision': out['decision']}, indent=1))
for r in rows:
    print(r['pair'], r['group'], r['song'], 'chose', r['chose'], r['winner'], '>', r['loser'], 'clip', r['clip_nd'],
          'higher', r['higher_nd_side'], r['basis'], r['tags'] or '')
