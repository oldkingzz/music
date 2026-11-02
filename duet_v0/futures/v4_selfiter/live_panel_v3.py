"""Duet live test page, version 3: live_panel_v2.py plus the round-trip latency calibration (run v5rt_1101, round 4).

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v3.py        # then open http://127.0.0.1:8766
  offline check, no sound device: .venv/bin/python runs_mini/v5rt_1101/tools/panel_v3_selftest_v1.py

Everything of v2 (modes rt / reactive / predict, the rt systems of the highest-numbered live_rt_systems_v<k>.json,
the blind A/B, history, ratings, the Bluetooth warning, file mode; the same run records: runs_mini/live/<HHMMSS>_<system
|blind>/ with run.wav and notes.json, each verdict appended to runs_mini/live/RUNS.jsonl), plus:
- 测延迟 (microphone mode): POST /api/calibrate opens the SAME duplex stream as a song (same devices, block size,
  latency setting and callback Session._cb) with a CalRun in place of the engine. CalRun.process(x) -> y like an
  engine: 0.5 s of silence, 4 count-in clicks, then 8 higher clicks at 100 bpm (latcal_v1.schedule); she sings a short
  'da' on each of the 8; it keeps the microphone input, sample-aligned with the output it wrote. At the end
  latcal_v1.estimate gives the round trip RT (median of her onset - click write time) or a short reason to retry. An ok
  result is kept per (input device, output device) pair for this page session (a failed retry keeps the earlier one);
  every attempt is appended to runs_mini/live/latcal_log.jsonl (append only) and its recording is saved in a new folder
  runs_mini/live/<HHMMSS>_latcal/ (cal.wav: L microphone, R clicks; cal.json). The monitor is off while it runs.
- A microphone run whose devices have a calibration tells every engine Li = max(0, RT - Lo), Lo = the stream's output
  latency (the round-trip error of every heard onset is then about 0; v2 told PortAudio's Li, which for an AirPods
  microphone is about 90 ms short, so her notes land on the wrong half beat). Without one it tells v2's numbers and the
  page asks to calibrate. The page and notes.json show the latency used ('lat').
- by_latency: if the list has "by_latency": {"threshold_ms": N, "system": name} and the run's devices have a calibrated
  round trip >= N ms, the rt choice 'auto' (the page's default) runs that system, otherwise the list's default; the page
  says which and why, and the rt blind A/B compares reactive r2_mix_v2 with the system 'auto' picks. Explicit choices
  still run what was chosen. (Session(root, rt=...) takes a list in memory for tests.)
- File mode is unchanged: no microphone and nothing to calibrate (Li 0, Lo = the stream's output latency, as v2).
- Device lists are not re-read while a run or a calibration is open.
live_panel_v2.py, live_panel_v2.html and every list file are unchanged; this file never writes a list.
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
import latcal_v1 as LC  # noqa: E402
import bench3  # noqa: E402
import live_mic as lm  # noqa: E402
from dsp import HOP, SR, write_wav  # noqa: E402

PORT = 8766
PAGE = 'live_panel_v3.html'
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
AUTO = 'auto'                                              # rt choice: the list's default or its by_latency system


def rt_list():
    """The highest-numbered live_rt_systems_v<k>.json next to this file (by_latency checked if present)."""
    found = []
    for p in glob.glob(os.path.join(HERE, 'live_rt_systems_v*.json')):
        m = re.search(r'live_rt_systems_v(\d+)\.json$', p)
        if m:
            found.append((int(m.group(1)), p))
    path = max(found)[1]
    with open(path) as fh:
        d = json.load(fh)
    assert d['default'] in d['systems'], path
    bl = d.get('by_latency')
    if bl is not None:
        assert isinstance(bl, dict) and isinstance(bl.get('system'), str), path
        assert isinstance(bl.get('threshold_ms'), (int, float)) and not isinstance(bl['threshold_ms'], bool), path
    d['file'] = os.path.basename(path)
    return d


RT = rt_list()


def rt_names(rt):
    """Every rt system a list can run: its systems and its by_latency system."""
    names = list(rt['systems'])
    bl = rt.get('by_latency')
    if bl and bl['system'] not in names:
        names.append(bl['system'])
    return names


def pick_rt(rt, cal):
    """The system the rt choice 'auto' runs: by_latency's system when the devices' calibrated round trip is >= its
    threshold, else the list's default. cal = the devices' calibration (None: not calibrated). -> (name, why)."""
    bl = rt.get('by_latency')
    if not bl:
        return rt['default'], ''
    thr = float(bl['threshold_ms'])
    if cal is None:
        return rt['default'], f'还没测延迟：用默认版（测出来 ≥ {thr:g} 毫秒才换大延迟版）'
    if cal['rt_ms'] >= thr:
        return bl['system'], f'来回延迟 {cal["rt_ms"]:.0f} 毫秒 ≥ {thr:g}：用大延迟版'
    return rt['default'], f'来回延迟 {cal["rt_ms"]:.0f} 毫秒 < {thr:g}：用默认版'


def label(name, mode, rt=None):
    """How a run's system is named on the page and in the history."""
    rt = RT if rt is None else rt
    if mode == 'rt':
        return rt['systems'].get(name, name)
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
            for name in rt_names(RT):
                self.rt[name] = LR.rt_build(name, train)
        except Exception as e:                             # shown on the page
            self.err = repr(e)
        self.ready.set()


MODELS = Models()


class CalRun:
    """The 测延迟 step as an engine for Session._cb: process(x) -> y writes the clicks of latcal_v1.schedule() and
    keeps the microphone input; rec sample i arrived in the callback that wrote output sample i."""

    def __init__(self, bpm=LC.BPM):
        self.sched = LC.schedule(SR, bpm)
        self.n = 0
        self.rec = []
        self.db = -120.0
        self.events = []
        self.Li = self.Lo = 0.0

    def process(self, x):
        x = np.asarray(x, np.float64)
        self.rec.append(x.astype(np.float32))
        y = LC.render_clicks(self.sched, self.n, len(x))
        self.n += len(x)
        self.db = float(10.0 * np.log10(np.mean(x * x) + 1e-12))
        return y

    def done(self):
        return self.n >= self.sched['n_total']

    def recording(self):
        return np.concatenate(self.rec) if self.rec else np.zeros(0, np.float32)

    def progress(self):
        """(phase, k, of) at the output sample being written: wait, count (count-in k of 4), sing (click k of 8), tail."""
        s, n = self.sched, self.n
        if n < s['count_in'][0]:
            return 'wait', 0, len(s['count_in'])
        if n < s['clicks'][0]:
            return 'count', sum(c <= n for c in s['count_in']), len(s['count_in'])
        k = sum(c <= n for c in s['clicks'])
        if n < s['clicks'][-1] + int(0.6 * SR):
            return 'sing', k, len(s['clicks'])
        return 'tail', k, len(s['clicks'])

    def estimate(self):
        return LC.estimate(self.recording(), SR, self.sched['clicks'])


class Session:
    def __init__(self, root, rt=None):
        self.root = root
        self.rt = RT if rt is None else rt
        self.lock = threading.Lock()
        self.eng, self.stream, self.run_dir, self.name, self.blind = None, None, None, None, False
        self.song = SONG
        self.state, self.msg, self.started = 'idle', '', None
        self.rec_in, self.rec_out, self.done = [], [], threading.Event()
        self.devs = ('', '')
        self.last = None                                   # the finished run waiting for a verdict
        self.monitor = 0.0
        self.cals = {}                                     # (input name, output name) -> latest ok calibration
        self.cal_last = None                               # the latest calibration attempt, ok or not
        self.cal_stream = (0.0, 0.0)                       # the stream's (Li, Lo) during that calibration
        self.lat = None                                    # what the current / last run's engine was told
        self.pick = None                                   # what 'auto' picks for the current / last run
        self.flags = []                                    # callback blocks with PortAudio status flags

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

    def _set_lat(self, eng, cal, stream_lat):
        """Tell the engine its latencies: with a calibration Li = max(0, RT - Lo), else the stream's (v2)."""
        Li_s, Lo_s = float(stream_lat[0]), float(stream_lat[1])
        if cal is not None:
            eng.Li, eng.Lo = max(0.0, float(cal['rt_s']) - Lo_s), Lo_s
        else:
            eng.Li, eng.Lo = Li_s, Lo_s
        self.lat = {'source': 'calibrated' if cal is not None else 'portaudio', 'Li_ms': round(eng.Li * 1000, 1),
                    'Lo_ms': round(eng.Lo * 1000, 1), 'rt_ms': None if cal is None else cal['rt_ms'],
                    'cal_when': None if cal is None else cal['when'],
                    'stream_Li_ms': round(Li_s * 1000, 1), 'stream_Lo_ms': round(Lo_s * 1000, 1)}

    def start(self, p, source=None, sim=None):
        """v2's start. source: audio fed offline through _cb (no sound device). sim = {'devs': (in name, out name),
        'latency': (Li, Lo)}: an offline run that uses those devices' calibration and stream latency as a live
        microphone run would (selftests)."""
        with self.lock:
            if self.state == 'running':
                raise ValueError('已经在跑了')
            if self.state == 'calibrating':
                raise ValueError('正在测延迟，等它测完')
            if self.state == 'await':
                raise ValueError('先给上一轮打分（或选“作废”）')
            if not MODELS.ready.is_set():
                raise ValueError('模型还在加载，稍等一会儿')
            if MODELS.err:
                raise ValueError('模型加载失败：' + MODELS.err)
            mode = p.get('mode', 'rt')
            if mode not in MODES:
                raise ValueError('没有这个模式：' + str(mode))
            choice = p.get('system', AUTO if mode == 'rt' else 'r2_mix_v2')
            blind = choice == 'blind'
            ab = blind and mode == 'rt'                    # A/B: reactive r2_mix_v2 or the rt system 'auto' picks
            if mode == 'rt' and not ab and choice != AUTO and choice not in self.rt['systems']:
                raise ValueError('没有这个实时系统：' + str(choice))
            if mode != 'rt' and not blind and choice not in CHOICES:
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
            from_file = p.get('input') == 'file'           # no mic: the phone file goes straight into Duet
            if click and from_file:
                raise ValueError('“跟节拍器唱”要用麦克风输入')
            dev = None
            if source is None:
                refresh_devices()
                try:
                    dev = (None if from_file else LB.pick('in', p.get('in') or ''), LB.pick('out', p.get('out') or ''))
                except SystemExit as e:
                    raise ValueError(str(e))
                import sounddevice as sd
                devs = (('文件：' + os.path.basename(song['wav']) + '（不开麦）') if from_file
                        else sd.query_devices(dev[0])['name'], sd.query_devices(dev[1])['name'])
            elif sim is not None:
                devs = (str(sim['devs'][0]), str(sim['devs'][1]))
            else:
                devs = ('selftest file', 'none')
            cal = None if from_file else self.cals.get(devs)
            auto, why = pick_rt(self.rt, cal)
            if ab:
                mode, name = random.choice((('reactive', 'r2_mix_v2'), ('rt', auto)))
            elif mode == 'rt':
                name = auto if choice == AUTO else choice
            else:
                name = random.choice(list(CHOICES)) if blind else choice
            if mode == 'rt' and name not in MODELS.rt:
                raise ValueError('这个实时系统没加载：' + str(name))
            eng = self._engine(mode, name, song, click, on_rt=click or ab)
            song.update({'mode': mode, 'click': click, 'engine': type(eng).__name__, 'ab': ab})
            self.song, self.monitor = song, monitor
            run_dir = os.path.join(self.root, time.strftime('%H%M%S') + ('_blind' if blind else '_' + name))
            os.makedirs(run_dir)                           # never reuse a folder
            self.eng, self.run_dir, self.name, self.blind = eng, run_dir, name, blind
            self.rec_in, self.rec_out, self.msg, self.flags = [], [], '', []
            self.done = threading.Event()
            self.started = time.time()
            self.pick = {'auto': auto, 'label': label(auto, 'rt', self.rt), 'why': why,
                         'used': choice == AUTO or ab, 'calibrated': cal is not None}
            self.lat = None
            if source is None and from_file:
                import sounddevice as sd
                import soundfile as sf
                her, sr = sf.read(os.path.join(LB.ABS, song['wav']), dtype='float32', always_2d=True)
                if sr != SR:
                    raise ValueError(f'{song["wav"]} 是 {sr} Hz，要 {SR} Hz')
                self.her, self.her_pos = her.mean(1), 0
                self.her_scale = 1.0 / max(float(np.percentile(np.abs(self.her), 99.95)), 1e-3)   # her at peak ~1
                self.devs = devs
                self.stream = sd.OutputStream(samplerate=SR, blocksize=HOP, channels=2, dtype='float32', latency='low',
                                              device=dev[1], callback=self._cb_file)
                eng.Li, eng.Lo = 0.0, self.stream.latency
                self.lat = {'source': 'file', 'Li_ms': 0.0, 'Lo_ms': round(float(eng.Lo) * 1000, 1), 'rt_ms': None,
                            'cal_when': None, 'stream_Li_ms': None, 'stream_Lo_ms': round(float(eng.Lo) * 1000, 1)}
                self.stream.start()
            elif source is None:
                import sounddevice as sd
                self.devs = devs
                self.stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=(1, 2), dtype='float32', latency='low',
                                        device=dev, callback=self._cb)
                self._set_lat(eng, cal, self.stream.latency)
                self.stream.start()
            else:
                self.devs = devs
                if sim is not None:
                    self._set_lat(eng, cal, sim['latency'])
                threading.Thread(target=self._feed, args=(source,), daemon=True).start()
            self.state = 'running'
            threading.Thread(target=self._watch, daemon=True).start()
            return self.status()

    def calibrate(self, p, source=None, sim=None):
        """测延迟: a CalRun on the song's duplex stream and callback. source / sim: offline, as in start()."""
        with self.lock:
            if self.state == 'running':
                raise ValueError('已经在跑了')
            if self.state == 'calibrating':
                raise ValueError('正在测延迟，等它测完')
            if self.state == 'await':
                raise ValueError('先给上一轮打分（或选“作废”）')
            if p.get('input') == 'file':
                raise ValueError('测延迟要用麦克风：把“输入”换成麦克风')
            cal = CalRun()
            dev = None
            if source is None:
                refresh_devices()
                try:
                    dev = (LB.pick('in', p.get('in') or ''), LB.pick('out', p.get('out') or ''))
                except SystemExit as e:
                    raise ValueError(str(e))
                import sounddevice as sd
                devs = (sd.query_devices(dev[0])['name'], sd.query_devices(dev[1])['name'])
            elif sim is not None:
                devs = (str(sim['devs'][0]), str(sim['devs'][1]))
            else:
                devs = ('selftest mic', 'none')
            run_dir = os.path.join(self.root, time.strftime('%H%M%S') + '_latcal')
            os.makedirs(run_dir)                           # never reuse a folder
            self.eng, self.run_dir, self.name, self.blind = cal, run_dir, None, False
            self.devs, self.monitor = devs, 0.0            # no monitor while the clicks play
            self.rec_in, self.rec_out, self.msg, self.flags = [], [], '', []
            self.done = threading.Event()
            self.started = time.time()
            self.lat, self.pick = None, None
            if source is None:
                import sounddevice as sd
                self.stream = sd.Stream(samplerate=SR, blocksize=HOP, channels=(1, 2), dtype='float32', latency='low',
                                        device=dev, callback=self._cb)
                self.cal_stream = tuple(float(v) for v in self.stream.latency)
                self.stream.start()
            else:
                self.cal_stream = tuple(float(v) for v in (sim or {}).get('latency', (0.0, 0.0)))
                threading.Thread(target=self._feed, args=(source,), daemon=True).start()
            self.state = 'calibrating'
            threading.Thread(target=self._watch, daemon=True).start()
            return self.status()

    def _cb(self, indata, outdata, frames, t, status):
        if status:
            self.flags.append(len(self.rec_in))
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
        if isinstance(self.eng, CalRun):
            self._finish_cal()
        else:
            self._finish()

    def stop(self):
        self.done.set()
        return self.status()

    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.msg = (self.msg + ' ' if self.msg else '') + '关闭声卡出错：' + repr(e)
            self.stream = None

    def _finish(self):
        with self.lock:
            self._close_stream()
            eng = self.eng
            s = eng.summary()
            s.update({'song': self.song['song'], 'key': self.song['key'], 'phone_file': self.song['file'],
                      'mode': self.song.get('mode'), 'click': self.song.get('click'), 'bpm': self.song['bpm'],
                      'engine': self.song.get('engine'), 'ab': self.song.get('ab'), 'rt_list': self.rt.get('file'),
                      'blind': self.blind, 'in': self.devs[0],
                      'out': self.devs[1], 'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started)),
                      'seconds': round(time.time() - self.started, 1), 'lat': self.lat, 'auto_pick': self.pick,
                      'stream_flag_blocks': len(self.flags), 'page': 'live_panel_v3'})
            if self.rec_in:
                st = np.stack([np.concatenate(self.rec_in), np.concatenate(self.rec_out)], 1)
                write_wav(os.path.join(self.run_dir, 'run.wav'), st)
            with open(os.path.join(self.run_dir, 'notes.json'), 'x') as fh:
                json.dump(s, fh, indent=1, default=float, ensure_ascii=False)
            lat = self.lat or {}
            self.last = {'run': os.path.basename(self.run_dir), 'system': self.name, 'blind': self.blind,
                         'label': label(self.name, self.song.get('mode'), self.rt), 'engine': self.song.get('engine'),
                         'song': self.song['song'], 'key': self.song['key'], 'mode': self.song.get('mode'),
                         'click': self.song.get('click'), 'bpm': self.song['bpm'],
                         'on_time': s.get('kept'), 'switched': s.get('corrected'),
                         'notes_answered': s['notes_answered'], 'sung': s['sung'],
                         'detect_ms_median': s['detect_ms_median'], 'in': self.devs[0], 'out': self.devs[1],
                         'started': s['started'], 'lat_source': lat.get('source'), 'Li_ms': lat.get('Li_ms'),
                         'Lo_ms': lat.get('Lo_ms'), 'rt_ms': lat.get('rt_ms'), 'page': 'v3'}
            self.state = 'await'

    def _finish_cal(self):
        with self.lock:
            self._close_stream()
            cal = self.eng
            complete = cal.done()
            try:
                r = cal.estimate()
            except Exception as e:                         # shown on the page
                r = {'ok': False, 'why': '估算出错：' + repr(e), 'rt_s': None, 'n_used': 0, 'spread_s': None}
            if not complete:
                r = dict(r, ok=False, why='没测完（停了或出错）：再测一次。' + (' ' + self.msg if self.msg else ''))
            now = time.time()
            rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)), 'in': self.devs[0],
                   'out': self.devs[1], 'stream_latency_ms': [round(v * 1000, 1) for v in self.cal_stream],
                   'ok': bool(r['ok']), 'why': r.get('why', ''),
                   'rt_ms': None if r.get('rt_s') is None else round(float(r['rt_s']) * 1000, 1),
                   'n_used': r.get('n_used', 0),
                   'spread_ms': None if r.get('spread_s') is None else round(float(r['spread_s']) * 1000, 1),
                   'offsets_ms': r.get('offsets_ms'), 'n_onsets': r.get('n_onsets'), 'floor_db': r.get('floor_db'),
                   'bpm': cal.sched['bpm'], 'complete': complete, 'stream_flag_blocks': len(self.flags),
                   'folder': os.path.basename(self.run_dir), 'page': 'live_panel_v3', 'latcal': 'latcal_v1'}
            if rec['ok']:
                self.cals[self.devs] = {'in': self.devs[0], 'out': self.devs[1], 'rt_s': float(r['rt_s']),
                                        'rt_ms': rec['rt_ms'], 'n_used': rec['n_used'], 'spread_ms': rec['spread_ms'],
                                        'when': time.strftime('%H:%M', time.localtime(now)),
                                        'stream_Li_ms': rec['stream_latency_ms'][0],
                                        'stream_Lo_ms': rec['stream_latency_ms'][1]}
            rec['kept'] = rec['ok']
            self.cal_last = rec
            x = cal.recording()
            if len(x):
                y = np.concatenate(self.rec_out)[:len(x)] if self.rec_out else np.zeros(len(x), np.float32)
                write_wav(os.path.join(self.run_dir, 'cal.wav'), np.stack([x, y], 1))
            with open(os.path.join(self.run_dir, 'cal.json'), 'x') as fh:
                json.dump(dict(rec, estimate=r, schedule=cal.sched), fh, indent=1, default=float, ensure_ascii=False)
            with open(os.path.join(self.root, 'latcal_log.jsonl'), 'a') as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
            self.state = 'idle'

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
               'rt_choices': self.rt['systems'], 'rt_default': self.rt['default'], 'rt_list': self.rt.get('file'),
               'by_latency': self.rt.get('by_latency'),
               'rt_labels': {n: label(n, 'rt', self.rt) for n in rt_names(self.rt)}, 'verdicts': VERDICTS,
               'run': os.path.basename(self.run_dir) if self.run_dir else None, 'blind': self.blind,
               'system': None if hide else self.name, 'devices': self.devs,
               'last': None if self.last is None else (dict(self.last, system=None, label=None, mode=None, engine=None)
                                                       if self.last['blind'] else self.last),
               'cals': list(self.cals.values()), 'cal_last': self.cal_last, 'lat': self.lat, 'pick': self.pick,
               'cal': None}
        if hide:
            out['song'] = dict(self.song, mode=None, engine=None)
        if isinstance(eng, CalRun):
            ph, k, of = eng.progress()
            out.update({'cal': {'phase': ph, 'k': k, 'of': of}, 'rows': [], 'n': 0, 'sung': 0, 'synced': False,
                        'beat': None, 'K': None, 'pitch': None, 'db': round(eng.db, 1),
                        'elapsed': round(time.time() - self.started, 1) if self.started else 0})
        elif eng is not None:
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
    """PortAudio reads the device list once; re-read it so headphones and mics plugged in later show up (never while a
    run or a calibration has a stream open)."""
    import sounddevice as sd
    if SESSION.state not in ('running', 'calibrating'):
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
            if p == '/api/calibrate':
                return self._send(200, SESSION.calibrate(body))
            if p == '/api/stop':
                return self._send(200, SESSION.stop())
            if p == '/api/verdict':
                return self._send(200, SESSION.verdict(body))
            return self._send(404, {'error': 'not found'})
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        except Exception as e:
            return self._send(500, {'error': repr(e)})


def main():
    if '--selftest' in sys.argv:
        sys.exit('offline check: .venv/bin/python runs_mini/v5rt_1101/tools/panel_v3_selftest_v1.py')
    os.makedirs(LB.OUT_ROOT, exist_ok=True)
    threading.Thread(target=MODELS.load, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'Duet live test page v3: http://127.0.0.1:{PORT}   (Ctrl-C stops)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.done.set()


if __name__ == '__main__':
    main()
