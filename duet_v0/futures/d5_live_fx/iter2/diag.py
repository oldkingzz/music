"""Why is the output silent while she sings? Classify each scored frame (vocadito ids) by her voicing / gate."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import clash, sim5  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location('fx2', HERE / 'fx.py')
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
lo, hi = map(int, sys.argv[1].split('-'))
C = {}
for i in range(lo, hi + 1):
    x, notes, tonic, tun = sim5.load_excerpt(i)
    fmin = 140.0 if i in sim5.FEMALE else 80.0
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    y, eng = (fx.DEFAULTS.update(kw), fx.run_open(x, tonic, aec=False, fmin=fmin, log=True))[1]
    L = np.array(eng.log, float)        # per block: k, f0voiced, ap, Lb, Ly, gate_open, gain, note, shift
    fy = sim5.frame_db(y); lvl = 10*np.log10(np.mean(10**(fy[fy > fy.max()-30]/10)))
    mo = clash.track_out(y, lvl - 30)
    fxx = sim5.frame_db(x); lx = 10*np.log10(np.mean(10**(fxx[fxx > fxx.max()-30]/10)))
    mi = clash.track_out(x, lx - 30)
    N = fx.StreamYin(24000, 90, 1000).N
    tcen = ((np.arange(len(mo)) + 1) * 240 - N / 2) / 24000
    for on, off, mm in notes:
        her = int(np.round(mm - tun))
        for j in np.nonzero((tcen >= on) & (tcen < off))[0]:
            b = min(int(tcen[j] * 24000 / 64), len(L) - 1)
            g = L[b, 5]; hv = j < len(mi) and not np.isnan(mi[j])
            v = mo[j]
            if np.isnan(v):
                key = 'sil_' + ('herV' if hv else 'herU') + ('_gateOpen' if g else '_gateClosed')
                if hv and g:
                    k1 = (j + 1) * 240; seg = y[max(k1 - N, 0):k1]
                    e = 10*np.log10(np.mean(seg**2)+1e-14)
                    key += '_quiet' if e < lvl - 30 else '_aper'
                    if e >= lvl - 30:
                        ea = L[b, 2]; ef = 24000 / 1 if False else np.nan
                        fh = fx.midi_to_hz(mi[j]); fe = L[b, 1]
                        if ea >= 0.45: key += '_engUnper'
                        elif np.isnan(fe): key += '_engWeak'
                        else:
                            r = 12 * np.log2(fe / fh)
                            key += '_engOct' if abs(r) > 6 else '_engOK'
                if hv and not g:
                    key += '_unvEng' if np.isnan(L[b,1]) else '_vEng'
            else:
                iv = abs(her - int(np.round(v - tun))) % 12
                cls = 'good' if iv in clash.GOOD else 'harsh' if iv in clash.HARSH else 'fourth'
                if cls != 'good':
                    # is her own pitch on the annotated note?
                    hm = mi[j] if hv else np.nan
                    ok = hv and int(np.round(hm - tun)) == her
                    key = cls + ('_herOnNote' if ok else '_herOffNote')
                    if ok:
                        s = eng.log[b][8]; key += f'_iv{iv}'
                else:
                    key = 'good'
            C[key] = C.get(key, 0) + 1
T = sum(C.values())
for k in sorted(C, key=lambda k: -C[k]):
    print(f'{k:40s} {100*C[k]/T:5.1f}%')
