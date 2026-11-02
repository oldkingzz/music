"""Rough printed-length estimate of the report drafts (read-only; prints only).

Strips HTML comments, skips blank lines and table separator rows, counts each table row as 1 line, each display-math block
as 2 lines, wraps other lines at COLS full-width columns (CJK = 1, ASCII = 0.5), headings x1.5, + 0.5 line per paragraph gap.
"""
import re
import sys

DIR = '/Users/san/claude_sandbox/music/duet_v0/runs_mini/v4_1101/report_drafts/'
COLS = float(sys.argv[1]) if len(sys.argv) > 1 else 46.0

for f in sys.argv[2:] or ['MORNING.md', 'TEAM.md', 'DIAGNOSIS_zh.md']:
    t = open(DIR + f).read()
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    t = re.sub(r'\$\$.*?\$\$', 'DISPLAYMATH', t, flags=re.S)
    tot, paras, prev_blank = 0.0, 0, True
    for line in t.split('\n'):
        s = line.strip()
        if not s:
            prev_blank = True
            continue
        if prev_blank:
            paras += 1
        prev_blank = False
        if re.match(r'^\|[-:| ]+\|$', s):
            continue
        if s == 'DISPLAYMATH':
            tot += 2
            continue
        if s.startswith('|'):
            tot += 1
            continue
        s = re.sub(r'\$[^$]*\$', 'xxxxxxxx', s).replace('**', '').replace('`', '')
        w = sum(1.0 if ord(c) > 0x2e80 else 0.5 for c in s)
        n = max(1, -(-w // COLS))
        tot += n * (1.5 if s.startswith('#') else 1)
    print(f, 'cols', COLS, 'est_lines', int(tot), 'paragraph_gaps', paras, 'total', int(tot + 0.5 * paras))
