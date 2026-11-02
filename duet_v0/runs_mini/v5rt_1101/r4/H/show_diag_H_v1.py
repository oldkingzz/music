"""Idea H: print W's and idea H's latency diagnostics side by side (flattened numbers of diag_lat_v1's per-setting
output, no row lists). Read-only; val only, tuned on val (no test split).

  show_diag_H_v1.py <W diag json> <H diag json> <setting>
"""
import json
import sys


def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ('hear_rows', 'acc', 'latency', 'bench_check'):
                continue
            out.update(flat(v, f'{pre}{k}.'))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[pre[:-1]] = d
    return out


def main():
    w_path, h_path, lat = sys.argv[1:4]
    with open(w_path) as fh:
        w = flat(json.load(fh)['settings'][lat])
    with open(h_path) as fh:
        h = flat(json.load(fh)['settings'][lat])
    for k in sorted(set(w) | set(h)):
        a, b = w.get(k), h.get(k)
        fa = '' if a is None else f'{a:.4f}' if isinstance(a, float) else str(a)
        fb = '' if b is None else f'{b:.4f}' if isinstance(b, float) else str(b)
        print(f'{k:70s} W {fa:>10s}   H {fb:>10s}')


if __name__ == '__main__':
    main()
