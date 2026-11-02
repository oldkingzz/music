"""Duet Studio: every offline effect on one recording, run in the background, results kept per source.

A source is a mono recording of her singing (a built-in test clip, a phrase recorded in the panel, or an
uploaded file). For each source the runner writes, in sessions/studio_<id>/:
  original.mp3, satb.mp3 (D1), own.mp3 (D2 fix), bach.mp3 / pop.mp3 / jazz.mp3 (D4), fx.mp3 (D5), duet.mp3
and status.json (one entry per effect: waiting / working / done / failed, seconds, error) plus info.json.
"""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DUET = HERE.parents[1]
PY = str(DUET / '.venv' / 'bin' / 'python')
SESS = HERE / 'sessions'
LAME = '/opt/homebrew/bin/lame'
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

EFFECTS = [
    ('satb', 'Four-part choir',
     'Duet transcribes her melody and writes alto, tenor and bass under it with Bach-style voice leading, '
     'sung by a synthetic choir.'),
    ('own', 'Her own choir',
     'The three harmony voices are her own recording, pitch-shifted to the chord tones, so the choir is all her.'),
    ('bach', 'Style: Bach', 'A four-part chorale harmonization of the same melody.'),
    ('pop', 'Style: Pop', 'Diatonic chords from a model trained on pop songs, as piano and bass.'),
    ('jazz', 'Style: Jazz', 'Seventh and ninth chords with ii-V and tritone substitutions, swung comping and walking bass.'),
    ('fx', 'Live harmonizer', 'The real-time effect: her voice shifted to the diatonic third below, run here on the recording.'),
    ('duet', 'Duet, beat by beat',
     'The original Duet: on every beat it guesses ahead and sings one note in its own voice '
     '(leaderboard policy; tempo found from her notes).'),
]
_LOCK = threading.Lock()
_RUNNING = set()


def builtin_sources():
    """Built-in clips (created once): a real singer (vocadito 11) and the synthetic Twinkle at 80 BPM."""
    out = []
    v = DUET / 'futures' / 'live_test' / 'orig_v11.wav'
    if v.exists():
        out.append(('vocadito11', 'Test singer: vocadito 11', 'Real amateur singer, 13 s. vocadito (Bittner et al.), CC BY 4.0.', v))
    t = DUET / 'futures' / 'live_test' / 'phone' / 'twinkle_80bpm_nocount.wav'
    src = DUET / 'futures' / 'live_test' / 'phone' / 'twinkle_C_80bpm_countin.wav'
    if not t.exists() and src.exists():
        x, sr = sf.read(src)
        sf.write(t, x[int(3.9 * sr):], sr)                 # the clip without its count-in clicks
    if t.exists():
        out.append(('twinkle', 'Twinkle Twinkle', 'Synthetic singer, C major, 80 BPM.', t))
    return out


def sources():
    """All sources, newest user recordings first, then the built-in clips."""
    items = []
    for d in sorted(SESS.glob('studio_*'), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = d / 'source.json'
        if meta.exists():
            m = json.load(open(meta))
            items.append(m)
    have = {m['id'] for m in items}
    for sid, title, note, wav in builtin_sources():
        if sid not in have:
            items.append({'id': sid, 'title': title, 'note': note, 'wav': str(wav), 'builtin': True})
    for m in items:
        d = SESS / f"studio_{m['id']}"
        m['status'] = json.load(open(d / 'status.json')) if (d / 'status.json').exists() else {}
        m['info'] = json.load(open(d / 'info.json')) if (d / 'info.json').exists() else {}
        m['files'] = {f.stem: f'studio_{m["id"]}/{f.name}' for f in d.glob('*.mp3')} if d.exists() else {}
        m['running'] = m['id'] in _RUNNING
    return items


def add_source(sid, title, note, wav):
    d = SESS / f'studio_{sid}'
    d.mkdir(parents=True, exist_ok=True)
    json.dump({'id': sid, 'title': title, 'note': note, 'wav': str(wav), 'builtin': False}, open(d / 'source.json', 'w'))
    return sid


def _set(d, key, **kw):
    p = d / 'status.json'
    st = json.load(open(p)) if p.exists() else {}
    st.setdefault(key, {}).update(kw)
    json.dump(st, open(p, 'w'))


def _run(cmd, cwd=DUET, timeout=600):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, env=dict(os.environ, PYTHONWARNINGS='ignore'))
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b'').decode(errors='ignore')[-500:] or f'exit {r.returncode}')


def process(sid, wav, force=False):
    """Run every effect on one source (sequentially, each in a subprocess)."""
    with _LOCK:
        if sid in _RUNNING:
            return
        _RUNNING.add(sid)
    d = SESS / f'studio_{sid}'
    d.mkdir(parents=True, exist_ok=True)
    try:
        if force and (d / 'status.json').exists():
            json.dump({}, open(d / 'status.json', 'w'))
        for k, _, _ in EFFECTS:
            if not (d / f'{k}.mp3').exists() or force:
                _set(d, k, state='waiting')
        x, sr = sf.read(wav)
        x = x.mean(1) if x.ndim > 1 else x
        src = d / 'source.wav'
        sf.write(src, x.astype(np.float32), sr)
        sf.write(d / 'original.wav', x.astype(np.float32), sr)
        subprocess.run([LAME, '--quiet', '-b', '128', str(d / 'original.wav'), str(d / 'original.mp3')], check=True)
        info = {'seconds': round(len(x) / sr, 1)}

        def step(key, fn):
            if (d / f'{key}.mp3').exists() and not force:
                return
            t0 = time.time()
            _set(d, key, state='working')
            try:
                fn()
                _set(d, key, state='done', seconds=round(time.time() - t0, 1))
            except Exception as e:  # noqa: BLE001
                _set(d, key, state='failed', error=str(e)[-400:])

        def d1():
            _run([PY, str(DUET / 'futures' / 'd1_record_harmonize' / 'run.py'), str(src), '-o', str(d / 'd1')])
            shutil.copyfile(d / 'd1' / 'choir.mp3', d / 'satb.mp3')
        step('satb', d1)
        timing = d / 'd1' / 'timing.json'
        if timing.exists():
            tj = json.load(open(timing))
            info.update(key=tj.get('key'), notes=len(tj.get('notes', [])))

        def own():
            code = ("from pathlib import Path; import core, choir; from make_demos import mp3\n"
                    f"x = core.load_audio(Path({str(src)!r})); r = choir.process(x, 'psola', 'human')\n"
                    f"mp3(Path({str(d / 'own.mp3')!r}), r['mix'])\n")
            _run([PY, '-c', code], cwd=DUET / 'futures' / 'd2_self_choir' / 'fix')
        step('own', own)

        def styles():
            _run([PY, str(HERE / 'studio_effects.py'), 'styles', str(src), str(timing), str(d)])
        for k in ('bach', 'pop', 'jazz'):
            if not (d / f'{k}.mp3').exists() or force:
                break
        else:
            k = None
        if k is not None:
            t0 = time.time()
            for k2 in ('bach', 'pop', 'jazz'):
                _set(d, k2, state='working')
            try:
                styles()
                for k2 in ('bach', 'pop', 'jazz'):
                    _set(d, k2, state='done', seconds=round(time.time() - t0, 1))
            except Exception as e:  # noqa: BLE001
                for k2 in ('bach', 'pop', 'jazz'):
                    _set(d, k2, state='failed', error=str(e)[-400:])
        if (d / 'styles.json').exists():
            sj = json.load(open(d / 'styles.json'))
            info['bpm'] = sj.get('bpm')
            info['styles'] = {k: sj.get(k, {}).get('no_clash') for k in ('bach', 'pop', 'jazz')}

        tonic = 0
        if info.get('key'):
            nm, *md = info['key'].split()
            tonic = NAMES.index(nm) if nm in NAMES else 0
            if md and md[0] == 'minor':
                tonic = (tonic + 3) % 12
        step('fx', lambda: _run([PY, str(HERE / 'studio_effects.py'), 'fx', str(src), str(tonic), str(d / 'fx.mp3')]))
        step('duet', lambda: _run([PY, str(HERE / 'studio_effects.py'), 'duet', str(src), str(timing), str(d / 'duet.mp3')]))
        if (d / 'duet.json').exists():
            info['duet'] = json.load(open(d / 'duet.json'))
        json.dump(info, open(d / 'info.json', 'w'))
    finally:
        with _LOCK:
            _RUNNING.discard(sid)


def start(sid, wav, force=False):
    threading.Thread(target=process, args=(sid, wav, force), daemon=True).start()


def warm_builtins():
    for sid, title, note, wav in builtin_sources():
        d = SESS / f'studio_{sid}'
        if not all((d / f'{k}.mp3').exists() for k, _, _ in EFFECTS):
            process(sid, str(wav))
