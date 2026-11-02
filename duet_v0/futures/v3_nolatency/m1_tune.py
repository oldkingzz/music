"""m1_filter tuning helper: val score without writing to the board. usage: m1_tune.py sysname"""
import importlib, sys, os, json
import bench3
name = sys.argv[1]
mod = importlib.import_module(f'systems.{name}')
sysm = mod.build(bench3.train_sets())
rows = {ds: bench3.score(sysm, songs) for ds, songs in bench3.eval_sets('val').items()}
h = sum(r['music'] for r in rows.values()) / len(rows)
env = {k: v for k, v in os.environ.items() if k.startswith('M1_') or k == 'DUET_HIST'}
print(f"{h:.4f} dbl{max(r['double'] for r in rows.values()):.3f}", env, ' '.join(f"{d}:{r['music']:.3f}/fit{r['fit']:.3f}/cons{r['cons']:.3f}/big{r['big']:.3f}/par{r['par']:.3f}" for d, r in rows.items()))
