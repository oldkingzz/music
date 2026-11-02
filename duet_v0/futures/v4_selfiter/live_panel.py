"""Duet live test page: phone speaker -> mic -> a bench3 system -> headphones, all from one local web page.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel.py        # then open http://127.0.0.1:8766
  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel.py --selftest

Pick a system (or 盲测: a random one, named only after the verdict), press 开始, play the phone file, press 停止
(or it stops by itself at the song's end), then rate the run. Each run is saved under
runs_mini/live/<HHMMSS>_<system|blind>/ (run.wav: L mic, R Duet; notes.json, mode 'x') and each verdict is
appended as one line to runs_mini/live/RUNS.jsonl. Only 127.0.0.1 is served; the mic is open only while a run
is going. The engine is live_bench3.Bench3Duet, the same code its --selftest checked.
"""
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
import bench3  # noqa: E402
import live_mic as lm  # noqa: E402
from dsp import HOP, SR, write_wav  # noqa: E402

PORT = 8766
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


class Models:
    """The three systems, built once in the background when the page starts."""

    def __init__(self):
        self.sys, self.err, self.ready = {}, None, threading.Event()

    def load(self):
        try:
            train = bench3.train_sets()
            for name in CHOICES:
                self.sys[name] = LB.build(name, train)
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
    def start(self, p, source=None):
        with self.lock:
            if self.state == 'running':
                raise ValueError('已经在跑了')
            if self.state == 'await':
                raise ValueError('先给上一轮打分（或选“作废”）')
            if not MODELS.ready.is_set():
                raise ValueError('模型还在加载，稍等几秒')
            if MODELS.err:
                raise ValueError('模型加载失败：' + MODELS.err)
            choice = p.get('system', 'r2_mix_v2')
            blind = choice == 'blind'
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
            mode = p.get('mode', 'predict')
            click = bool(p.get('click')) or song['song'] == 'free'
            if click and p.get('input') == 'file':
                raise ValueError('“跟节拍器唱”要用麦克风输入')
            args = (MODELS.sys[name], name, lm.KEYS[song['key']], song['meter'], song['beats'], song['bpm'])
            if mode == 'predict':
                eng = LP.PredictDuet(*args, first_beat=song['first_beat'], monitor=0.0, log=lambda s: None,
                                     count_in=4 if click else 0, click_all=click)
            else:
                eng = LB.Bench3Duet(*args, first_beat=song['first_beat'], monitor=0.0, log=lambda s: None)
            song['mode'], song['click'] = mode, click
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
                      'blind': self.blind, 'in': self.devs[0],
                      'out': self.devs[1], 'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started)),
                      'seconds': round(time.time() - self.started, 1)})
            if self.rec_in:
                st = np.stack([np.concatenate(self.rec_in), np.concatenate(self.rec_out)], 1)
                write_wav(os.path.join(self.run_dir, 'run.wav'), st)
            with open(os.path.join(self.run_dir, 'notes.json'), 'x') as fh:
                json.dump(s, fh, indent=1, default=float, ensure_ascii=False)
            self.last = {'run': os.path.basename(self.run_dir), 'system': self.name, 'blind': self.blind,
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
        out = {'state': self.state, 'msg': self.msg, 'models_ready': MODELS.ready.is_set(), 'models_err': MODELS.err,
               'song': self.song, 'songs': SONGS, 'keys': list(lm.PC_NAMES), 'choices': CHOICES, 'verdicts': VERDICTS,
               'run': os.path.basename(self.run_dir) if self.run_dir else None, 'blind': self.blind,
               'system': None if (self.blind and self.state != 'idle') else self.name,
               'devices': self.devs, 'last': self.last}
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
                             'act': e.get('action')})
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
        r['label'] = CHOICES.get(r.get('system'), r.get('system'))
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
                with open(os.path.join(HERE, 'live_panel.html'), 'rb') as fh:
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
    """The phone file fed through the page's own session code (no sound card), into runs_mini/live/selftest/."""
    import soundfile as sf
    MODELS.load()
    root = os.path.join(LB.OUT_ROOT, 'selftest')
    os.makedirs(root, exist_ok=True)
    s = Session(root)
    files = {'twinkle': 'twinkle_C_80bpm_countin.wav', 'v11': 'test_her_v11_countin.wav'}
    for choice, song, mode in (('r2_mix_v2', 'twinkle', 'predict'), ('blind', 'twinkle', 'predict'),
                               ('r2_mix_v2', 'v11', 'predict'), ('r2_mix_v2', 'twinkle', 'reactive')):
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
        rec = s.verdict({'run': st['run'], 'verdict': '作废', 'note': 'selftest'})
        ev = s.eng.events
        cons = sum(e['duet'] >= 0 and abs(e['her_heard'] - e['duet']) % 12 in CONS for e in ev)
        print(f"{choice:10s} {song:8s} {mode:8s} on time {rec['on_time']} switched {rec['switched']}  "
              f"key {rec['key']}  run {rec['run']}  system {rec['system']}  answered {rec['notes_answered']}  "
              f"sung {rec['sung']}  consonant with heard {cons}  "
              f"rows shown {len(st['rows'])}  wav {os.path.exists(os.path.join(root, rec['run'], 'run.wav'))}", flush=True)
        time.sleep(1.1)                                    # run folders are named by the second
    print('history rows', len(history(root)))


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    os.makedirs(LB.OUT_ROOT, exist_ok=True)
    threading.Thread(target=MODELS.load, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'Duet live test page: http://127.0.0.1:{PORT}   (Ctrl-C stops)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.done.set()


if __name__ == '__main__':
    main()
