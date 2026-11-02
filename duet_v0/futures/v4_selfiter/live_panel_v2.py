"""Duet live test page, version 2: live_panel.py plus the latency-aware real-time systems of run v5rt_1101.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v2.py        # then open http://127.0.0.1:8766
  offline check, no sound device: .venv/bin/python runs_mini/v5rt_1101/tools/panel_v2_selftest_v1.py

Same page, controls and run records as live_panel.py (runs_mini/live/<HHMMSS>_<system|blind>/ with run.wav and
notes.json; each verdict appended to runs_mini/live/RUNS.jsonl), plus:
- mode 'rt' (the default): live_rt.RTDuet runs a futures/v5_rt/rt_systems system: plan points every half beat,
  decided as late as possible and written early by the output latency; hear points as the tracker answers her notes.
  The systems offered come from the highest-numbered futures/v4_selfiter/live_rt_systems_v<k>.json
  ({"default": name, "systems": {name: label}}), so a later round adds a system with a new file, not an edit.
- 盲测 in mode 'rt' is an A/B test: reactive r2_mix_v2 or the default rt system, both on RTDuet (same tracker, voice and
  click), named only after the verdict.
- reactive with the click (singing along with Duet's count-in) runs r2_mix_v2 on RTDuet, so the click sounds
  (Bench3Duet has none); without the click reactive stays Bench3Duet, the engine Vincent heard on 2026-11-01.
live_panel.py and live_panel.html are unchanged.
"""
import glob
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_bench3 as LB  # noqa: E402  (puts duet_beat, live, panel and v3 on sys.path)
import live_predict as LP  # noqa: E402
import live_rt as LR  # noqa: E402  (puts futures/v5_rt on sys.path)
import bench3  # noqa: E402
import live_mic as lm  # noqa: E402
from dsp import HOP, SR, write_wav  # noqa: E402

PORT = 8766
PAGE = 'live_panel_v2.html'
CHOICES = {'r2_mix_v2': '新模型 r2_mix_v2', 'm1_filter_v5lo': '老模型 m1', 'third_below': '三度下方（基线）'}
# The phone files. Key, tempo and her first note's beat were measured from the files themselves on 2026-11-01
# (countin.CountIn on the count-in clicks; Krumhansl major profile on her notes): Twinkle C major 80.0 bpm,
# first note on beat 1 (0.013); vocadito 11 A major (r 0.86) 122.45 bpm, first note on beat 1 (0.041), ~26 beats.
SONGS = {
    'twinkle': {'song': 'twinkle', 'label': 'Twinkle（C 大调，80 拍）', 'key': 'C', 'meter': 4, 'beats': 48,
                'bpm': 80.0, 'first_beat': 0.0, 'file': 'futures/live_test/phone/twinkle_C_80bpm_countin.mp3',
                'wav': 'futures/live_test/phone/twinkle_C_80bpm_countin.wav'},
    'v11': {'song': 'v11', 'label': '真人演唱 v11（A 大调，122 拍）', 'key': 'A', 'meter': 4, 'beats': 28,
            'bpm': 122.45, 'first_beat': 0.0, 'file': 'futures/live_test/phone/test_her_v11_countin.mp3',
            'wav': 'futures/live_test/phone/test_her_v11_countin.wav'},
    'free': {'song': 'free', 'label': '我自己唱（跟 Duet 的节拍器）', 'key': 'C', 'meter': 4, 'beats': 64,
             'bpm': 80.0, 'first_beat': 0.0, 'file': '', 'wav': ''},
}
# The headphone mix copies make_demos3.render (the offline demos): her voice at peak 1 and Duet's reed at peak
# 0.42, L = 0.8 her + 0.35 Duet, R = 0.45 her + 0.75 Duet, then about 0.9 overall. In mic mode she is heard in
# the room (or through the monitor), so only Duet's side of that mix is played.
MIX_L, MIX_R, GAIN = (0.8, 0.35), (0.45, 0.75), 0.9
MON_GAIN = 8.0                                             # page slider 0..1 -> mic gain 0..8 in the headphones
SONG = SONGS['twinkle']
VERDICTS = ('好', '一般', '差', '作废')
CONS = {3, 4, 7, 8, 9}
MODES = ('rt', 'reactive', 'predict')


def rt_list():
    """The highest-numbered live_rt_systems_v<k>.json next to this file."""
    found = []
    for p in glob.glob(os.path.join(HERE, 'live_rt_systems_v*.json')):
        m = re.search(r'live_rt_systems_v(\d+)\.json$', p)
        if m:
            found.append((int(m.group(1)), p))
    path = max(found)[1]
    with open(path) as fh:
        d = json.load(fh)
    assert d['default'] in d['systems'], path
    d['file'] = os.path.basename(path)
    return d


RT = rt_list()


def label(name, mode):
    """How a run's system is named on the page and in the history."""
    if mode == 'rt':
        return RT['systems'].get(name, name)
    base = CHOICES.get(name, name)
    return {'reactive': '被动 · ', 'predict': '旧预判 · '}.get(mode, '') + base


class Reactive:
    """bench_rt's 'reactive:<v3 system>' adapter (futures/v5_rt/bench_rt.py Reactive): plan() never changes Duet's
    note, hear() is the bench3 system's step(). It lets r2_mix_v2 run on RTDuet (click count-in, A/B test)."""

    def __init__(self, v3):
        self.v3 = v3

    def start(self, meta):
        self.v3.start(meta)

    def plan(self, q, heard):
        return None

    def hear(self, i, onset, midi, prev_dur):
        return int(self.v3.step(i, onset, midi, prev_dur))


class Models:
    """The bench3 systems (reactive / predict) and the rt systems of RT, built once in the background at start."""

    def __init__(self):
        self.sys, self.rt, self.err, self.ready = {}, {}, None, threading.Event()

    def load(self):
        try:
            train = bench3.train_sets()
            for name in CHOICES:
                self.sys[name] = LB.build(name, train)
            for name in RT['systems']:
                self.rt[name] = LR.rt_build(name, train)
        except Exception as e:                             # shown on the page
            self.err = repr(e)
        self.ready.set()


MODELS = Models()


class Session:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.eng, self.stream, self.run_dir, self.name, self.blind = None, None, None, None, False
        self.song = SONG
        self.state, self.msg, self.started = 'idle', '', None
        self.rec_in, self.rec_out, self.done = [], [], threading.Event()
        self.devs = ('', '')
        self.last = None                                   # the finished run waiting for a verdict

    # ---------------------------------------------------------------- run
    def _engine(self, mode, name, song, click, on_rt):
        """The engine for one run; on_rt puts a reactive bench3 system on RTDuet (click, A/B)."""
        args = (lm.KEYS[song['key']], song['meter'], song['beats'], song['bpm'])
        common = dict(first_beat=song['first_beat'], monitor=0.0, log=lambda s: None)
        rtkw = dict(common, count_in=4 if click else 0, click_all=click)
        if mode == 'rt':
            return LR.RTDuet(MODELS.rt[name], name, *args, **rtkw)
        if mode == 'predict':
            return LP.PredictDuet(MODELS.sys[name], name, *args, **rtkw)
        if on_rt:
            return LR.RTDuet(Reactive(MODELS.sys[name]), name, *args, **rtkw)
        return LB.Bench3Duet(MODELS.sys[name], name, *args, **common)

    def start(self, p, source=None):
        with self.lock:
            if self.state == 'running':
                raise ValueError('已经在跑了')
            if self.state == 'await':
                raise ValueError('先给上一轮打分（或选“作废”）')
            if not MODELS.ready.is_set():
                raise ValueError('模型还在加载，稍等一会儿')
            if MODELS.err:
                raise ValueError('模型加载失败：' + MODELS.err)
            mode = p.get('mode', 'rt')
            if mode not in MODES:
                raise ValueError('没有这个模式：' + str(mode))
            choice = p.get('system', RT['default'] if mode == 'rt' else 'r2_mix_v2')
            blind = choice == 'blind'
            ab = blind and mode == 'rt'                    # A/B: reactive r2_mix_v2 or the default rt system
            if ab:
                mode, name = random.choice((('reactive', 'r2_mix_v2'), ('rt', RT['default'])))
            elif mode == 'rt':
                name = choice
                if name not in RT['systems']:
                    raise ValueError('没有这个实时系统：' + str(choice))
            else:
                name = random.choice(list(CHOICES)) if blind else choice
                if name not in CHOICES:
                    raise ValueError('没有这个系统：' + str(choice))
            if p.get('song', 'twinkle') not in SONGS:
                raise ValueError('没有这个手机文件：' + str(p.get('song')))
            song = dict(SONGS[p.get('song', 'twinkle')])
            if p.get('key'):
                if p['key'] not in lm.KEYS:
                    raise ValueError('不认识这个调：' + str(p['key']))
                song['key'] = p['key']
            if p.get('bpm'):
                song['bpm'] = min(max(float(p['bpm']), 40.0), 200.0)
            monitor = min(max(float(p.get('monitor', 0) or 0), 0.0), 1.0)
            click = bool(p.get('click')) or song['song'] == 'free'
            if click and p.get('input') == 'file':
                raise ValueError('“跟节拍器唱”要用麦克风输入')
            eng = self._engine(mode, name, song, click, on_rt=click or ab)
            song.update({'mode': mode, 'click': click, 'engine': type(eng).__name__, 'ab': ab})
            self.song, self.monitor = song, monitor
            dev = None
            from_file = p.get('input') == 'file'           # no mic: the phone file goes straight into Duet
            if source is None:
                refresh_devices()
                try:
                    dev = (None if from_file else LB.pick('in', p.get('in') or ''), LB.pick('out', p.get('out') or ''))
                except SystemExit as e:
                    raise ValueError(str(e))
            run_dir = os.path.join(self.root, time.strftime('%H%M%S') + ('_blind' if blind else '_' + name))
            os.makedirs(run_dir)                           # never reuse a folder
            self.eng, self.run_dir, self.name, self.blind = eng, run_dir, name, blind
            self.rec_in, self.rec_out, self.msg = [], [], ''
            self.done = threading.Event()
            self.started = time.time()
            if source is None and from_file:
                import sounddevice as sd
                import soundfile as sf
                her, sr = sf.read(os.path.join(LB.ABS, song['wav']), dtype='float32', always_2d=True)
                if sr != SR:
                    raise ValueError(f'{song["wav"]} 是 {sr} Hz，要 {SR} Hz')
                self.her, self.her_pos = her.mean(1), 0
                self.her_scale = 1.0 / max(float(np.percentile(np.abs(self.her), 99.95)), 1e-3)   # her at peak ~1
                self.devs = ('文件：' + os.path.basename(song['wav']) + '（不开麦）', sd.query_devices(dev[1])['name'])
                self.stream = sd.OutputStream(samplerate=SR, blocksize=HOP, channels=2, dtype='float32', latency='low',
                                              device=dev[1], callback=self._cb_file)
                eng.Li, eng.Lo = 0.0, self.stream.latency
                self.stream.start()
            elif source is None:
                import sounddevice as sd
                self.devs = (sd.query_devices(dev[0])['name'], sd.query_devices(dev[1])['name'])
                self.stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=(1, 2), dtype='float32', latency='low',
                                        device=dev, callback=self._cb)
                eng.Li, eng.Lo = self.stream.latency
                self.stream.start()
            else:
                self.devs = ('selftest file', 'none')
                threading.Thread(target=self._feed, args=(source,), daemon=True).start()
            self.state = 'running'
            threading.Thread(target=self._watch, daemon=True).start()
            return self.status()

    def _cb(self, indata, outdata, frames, t, status):
        x = indata[:, 0].copy()
        try:
            y = self.eng.process(x) if frames == HOP else np.zeros(frames)
        except Exception as e:                             # stop the run, keep what was recorded
            self.msg = '引擎出错：' + repr(e)
            y = np.zeros(frames)
            self.done.set()
        y = np.asarray(y, np.float64)
        mon = np.clip(MON_GAIN * self.monitor * x, -0.9, 0.9)          # the room mic is quiet (about -35 dB rms)
        outdata[:, 0] = np.clip(GAIN * MIX_L[1] * y + mon, -1, 1)
        outdata[:, 1] = np.clip(GAIN * MIX_R[1] * y + mon, -1, 1)
        self.rec_in.append(x.astype(np.float32))
        self.rec_out.append(np.clip(y, -1, 1).astype(np.float32))
        if self.eng.done():
            self.done.set()

    def _cb_file(self, outdata, frames, t, status):
        """No-mic mode: her voice from the phone file goes into Duet and into the headphones with Duet's voice."""
        a = self.her_pos
        x = self.her[a:a + frames]
        if len(x) < frames:
            x = np.pad(x, (0, frames - len(x)))
            self.done.set()
        self.her_pos = a + frames
        try:
            y = self.eng.process(x) if frames == HOP else np.zeros(frames)
        except Exception as e:
            self.msg = '引擎出错：' + repr(e)
            y = np.zeros(frames)
            self.done.set()
        h = self.her_scale * np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        outdata[:, 0] = np.clip(GAIN * (MIX_L[0] * h + MIX_L[1] * y), -1, 1)
        outdata[:, 1] = np.clip(GAIN * (MIX_R[0] * h + MIX_R[1] * y), -1, 1)
        self.rec_in.append(np.asarray(x, np.float32))
        self.rec_out.append(np.clip(y, -1, 1).astype(np.float32))
        if self.eng.done():
            self.done.set()

    def _feed(self, x):
        for b in range(len(x) // HOP):
            if self.done.is_set():
                break
            self._cb(x[b * HOP:(b + 1) * HOP, None], np.zeros((HOP, 2), np.float32), HOP, None, None)
        self.done.set()

    def _watch(self):
        self.done.wait()
        self._finish()

    def stop(self):
        self.done.set()
        return self.status()

    def _finish(self):
        with self.lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    self.msg = (self.msg + ' ' if self.msg else '') + '关闭声卡出错：' + repr(e)
                self.stream = None
            eng = self.eng
            s = eng.summary()
            s.update({'song': self.song['song'], 'key': self.song['key'], 'phone_file': self.song['file'],
                      'mode': self.song.get('mode'), 'click': self.song.get('click'), 'bpm': self.song['bpm'],
                      'engine': self.song.get('engine'), 'ab': self.song.get('ab'), 'rt_list': RT['file'],
                      'blind': self.blind, 'in': self.devs[0],
                      'out': self.devs[1], 'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started)),
                      'seconds': round(time.time() - self.started, 1)})
            if self.rec_in:
                st = np.stack([np.concatenate(self.rec_in), np.concatenate(self.rec_out)], 1)
                write_wav(os.path.join(self.run_dir, 'run.wav'), st)
            with open(os.path.join(self.run_dir, 'notes.json'), 'x') as fh:
                json.dump(s, fh, indent=1, default=float, ensure_ascii=False)
            self.last = {'run': os.path.basename(self.run_dir), 'system': self.name, 'blind': self.blind,
                         'label': label(self.name, self.song.get('mode')), 'engine': self.song.get('engine'),
                         'song': self.song['song'], 'key': self.song['key'], 'mode': self.song.get('mode'),
                         'click': self.song.get('click'), 'bpm': self.song['bpm'],
                         'on_time': s.get('kept'), 'switched': s.get('corrected'),
                         'notes_answered': s['notes_answered'], 'sung': s['sung'],
                         'detect_ms_median': s['detect_ms_median'], 'in': self.devs[0], 'out': self.devs[1],
                         'started': s['started']}
            self.state = 'await'

    def verdict(self, p):
        with self.lock:
            if self.state != 'await' or not self.last or p.get('run') != self.last['run']:
                raise ValueError('没有在等评价的这一轮')
            v = p.get('verdict')
            if v not in VERDICTS:
                raise ValueError('评价只能是 ' + ' / '.join(VERDICTS))
            rec = dict(self.last, verdict=v, note=str(p.get('note') or '')[:500],
                       rated=time.strftime('%Y-%m-%d %H:%M:%S'))
            with open(os.path.join(self.root, 'RUNS.jsonl'), 'a') as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.last, self.state = None, 'idle'
            return rec

    # ---------------------------------------------------------------- what the page shows
    def status(self):
        eng = self.eng
        hide = self.blind and self.state != 'idle'
        out = {'state': self.state, 'msg': self.msg, 'models_ready': MODELS.ready.is_set(), 'models_err': MODELS.err,
               'song': self.song, 'songs': SONGS, 'keys': list(lm.PC_NAMES), 'choices': CHOICES,
               'rt_choices': RT['systems'], 'rt_default': RT['default'], 'verdicts': VERDICTS,
               'run': os.path.basename(self.run_dir) if self.run_dir else None, 'blind': self.blind,
               'system': None if hide else self.name, 'devices': self.devs,
               'last': None if self.last is None else (dict(self.last, system=None, label=None, mode=None, engine=None)
                                                       if self.last['blind'] else self.last)}
        if hide:
            out['song'] = dict(self.song, mode=None, engine=None)
        if eng is not None:
            ev = list(eng.events)
            rows = []
            for e in ev[-60:]:
                h, m = e['duet'], e['her_heard']
                iv = None if h < 0 else abs(m - h) % 12
                kind = 'silent' if h < 0 else ('double' if iv == 0 else ('cons' if iv in CONS else 'clash'))
                bar, beat = divmod(int(np.floor(e['onset_beats'] + 1e-6)), self.song['meter'])
                rows.append({'pos': f'{bar + 1}.{beat + 1}', 'her': lm.note_name(m),
                             'duet': lm.note_name(h) if h >= 0 else '—', 'kind': kind, 'ms': e['detect_ms'],
                             'act': e.get('action') or e.get('act')})
            heard = eng.heard()
            pitch, db = None, None
            if eng.ft_t:
                db = float(eng.ft.db[-1])
                if eng.ft_v[-1]:
                    pitch = lm.note_name(int(round(float(eng.ft_m[-1]))))
            out.update({'rows': rows, 'n': len(ev), 'sung': sum(e['duet'] >= 0 for e in ev), 'synced': eng.t0 is not None,
                        'beat': None if eng.t0 is None else round((heard - eng.t0) / eng.T_b, 2), 'K': eng.K,
                        'pitch': pitch, 'db': None if db is None else round(db, 1),
                        'elapsed': round(time.time() - self.started, 1) if self.started else 0})
        return out


def refresh_devices():
    """PortAudio reads the device list once; re-read it so headphones and mics plugged in later show up."""
    import sounddevice as sd
    if SESSION.state != 'running':
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass


def devices():
    import sounddevice as sd
    refresh_devices()
    ins, outs = [], []
    for d in sd.query_devices():
        if d['max_input_channels'] > 0:
            ins.append(d['name'])
        if d['max_output_channels'] > 0:
            outs.append(d['name'])
    return {'in': ins, 'out': outs}


def history(root):
    path = os.path.join(root, 'RUNS.jsonl')
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        r['label'] = r.get('label') or label(r.get('system'), r.get('mode'))
        r['has_wav'] = os.path.exists(os.path.join(root, r['run'], 'run.wav'))
    return rows[::-1]


SESSION = Session(LB.OUT_ROOT)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=float).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p in ('/', '/index.html'):
                with open(os.path.join(HERE, PAGE), 'rb') as fh:
                    return self._send(200, fh.read(), 'text/html; charset=utf-8')
            if p == '/api/state':
                return self._send(200, SESSION.status())
            if p == '/api/devices':
                return self._send(200, devices())
            if p == '/api/runs':
                return self._send(200, history(SESSION.root))
            m = re.fullmatch(r'/rec/([0-9]{6}_[A-Za-z0-9_]+)/run\.wav', unquote(p))
            if m:
                f = os.path.join(SESSION.root, m.group(1), 'run.wav')
                if os.path.exists(f):
                    with open(f, 'rb') as fh:
                        return self._send(200, fh.read(), 'audio/wav')
            return self._send(404, {'error': 'not found'})
        except Exception as e:
            return self._send(500, {'error': repr(e)})

    def do_POST(self):
        p = urlparse(self.path).path
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
            if p == '/api/start':
                return self._send(200, SESSION.start(body))
            if p == '/api/stop':
                return self._send(200, SESSION.stop())
            if p == '/api/verdict':
                return self._send(200, SESSION.verdict(body))
            return self._send(404, {'error': 'not found'})
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        except Exception as e:
            return self._send(500, {'error': repr(e)})


def selftest():
    """The phone files fed through the page's own session code (no sound card) into runs_mini/live/selftest_v2/:
    every rt system on Twinkle, the default on v11, the rt A/B blind test and reactive on Twinkle; all rated 作废."""
    import soundfile as sf
    MODELS.load()
    assert MODELS.err is None, MODELS.err
    root = os.path.join(LB.OUT_ROOT, 'selftest_v2')
    os.makedirs(root, exist_ok=True)
    s = Session(root)
    files = {'twinkle': 'twinkle_C_80bpm_countin.wav', 'v11': 'test_her_v11_countin.wav'}
    runs = [(name, 'twinkle', 'rt') for name in RT['systems']]
    runs += [(RT['default'], 'v11', 'rt'), ('blind', 'twinkle', 'rt'), ('r2_mix_v2', 'twinkle', 'reactive')]
    for choice, song, mode in runs:
        x, sr = sf.read(os.path.join(LB.ABS, 'futures', 'live_test', 'phone', files[song]), dtype='float32', always_2d=True)
        assert sr == SR
        st = s.start({'system': choice, 'song': song, 'mode': mode}, source=x.mean(1))
        assert st['state'] == 'running'
        t0 = time.time()
        while s.state != 'await' and time.time() - t0 < 120:
            time.sleep(0.05)
        st = s.status()
        assert st['state'] == 'await', st['state']
        assert (st['system'] is None) == (choice == 'blind')
        assert (st['last']['label'] is None) == (choice == 'blind')
        rec = s.verdict({'run': st['run'], 'verdict': '作废', 'note': 'selftest v2'})
        ev = s.eng.events
        cons = sum(e['duet'] >= 0 and abs(e['her_heard'] - e['duet']) % 12 in CONS for e in ev)
        acts = sorted({r['act'] for r in st['rows'] if r['act']})
        print(f"{choice:18s} {song:8s} {mode:8s} -> {rec['mode']:8s} {rec['engine']:11s} {rec['label']}: "
              f"on time {rec['on_time']} switched {rec['switched']} answered {rec['notes_answered']} sung {rec['sung']} "
              f"consonant with heard {cons} rows {len(st['rows'])} acts {acts} "
              f"wav {os.path.exists(os.path.join(root, rec['run'], 'run.wav'))}", flush=True)
        time.sleep(1.1)                                    # run folders are named by the second
    print('history rows', len(history(root)), 'rt list', RT['file'], 'default', RT['default'])


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    os.makedirs(LB.OUT_ROOT, exist_ok=True)
    threading.Thread(target=MODELS.load, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'Duet live test page v2: http://127.0.0.1:{PORT}   (Ctrl-C stops)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.done.set()


if __name__ == '__main__':
    main()
