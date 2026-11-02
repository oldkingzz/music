"""Duet test panel: one local web page to run Duet on this Mac without the lead in the loop.

  cd project/duet_v0 && .venv/bin/python futures/panel/duet_panel.py      # then open http://127.0.0.1:8765

Live:   live harmonizer (D5 engine, iter2): mic -> diatonic third below -> output, with meters, key, gains,
        gate preset and an optional monitor of her voice in headphones. Every live run is recorded.
Studio: every offline effect on a clip, a recorded phrase or an uploaded file (studio.py).
Only 127.0.0.1 is served. The mic and output open only when you press a start/record button.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = Path(__file__).resolve().parent
DUET = HERE.parents[1]                                   # project/duet_v0
PY = str(DUET / '.venv' / 'bin' / 'python')
SESS = HERE / 'sessions'
SESS.mkdir(exist_ok=True)
sys.path.insert(0, str(DUET / 'futures' / 'd5_live_fx' / 'iter2'))
import fx  # noqa: E402
import studio  # noqa: E402

SR, B = 48000, 128
_DB = {}


def duet_beat():
    """The original real-time Duet (duet_beat/live_mic.py), imported on first use."""
    if not _DB:
        for p in (DUET / 'live', DUET, DUET / 'duet_beat'):
            sys.path.insert(0, str(p))
        import live_mic
        import planner
        import songs
        import audio
        import theory
        import metrics
        import policy_live
        _DB.update(policy_live=policy_live, live_mic=live_mic, planner=planner, songs=songs, audio=audio, theory=theory, metrics=metrics,
                   models={}, res=json.load(open(DUET / 'duet_beat' / 'out' / 'results.json')))
    return _DB


NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
GATES = {
    'normal': {'drop_db': 16, 'onset_db': 26, 'f0_tau': 0.015, 'mode': 'follow'},               # iter2 chosen v2
    'lenient': {'drop_db': None, 'onset_db': 40, 'f0_tau': 0.015, 'mode': 'follow', 'ap_open': 0.45, 'ap_stay': 0.65},
}
TEST_CLIPS = {'test_her_v11.mp3': DUET / 'futures' / 'live_test' / 'phone' / 'test_her_v11.mp3'}


def note_name(m):
    if m is None or not np.isfinite(m):
        return ''
    m = int(round(m))
    return f'{NAMES[m % 12]}{m // 12 - 1}'


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def devices(refresh=False):
    if refresh and LIVE.stream is None:              # PortAudio lists devices once per init: re-init to see new ones
        sd._terminate()
        sd._initialize()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({'index': i, 'name': d['name'], 'in': d['max_input_channels'], 'out': d['max_output_channels'],
                    'sr': int(d['default_samplerate']), 'iphone': 'iPhone' in d['name']})
    return out


def pick_default(devs):
    ins = [d for d in devs if d['in'] > 0 and not d['iphone']]
    outs = [d for d in devs if d['out'] > 0]
    def first(lst, words):
        for w in words:
            for d in lst:
                if w in d['name']:
                    return d['index']
        return lst[0]['index'] if lst else None
    return {'in': first(ins, ['MacBook', '麦克风', 'Microphone']),
            'out': first(outs, ['外置耳机', 'External Headphones', 'AirPods', 'Headphones', '扬声器', 'Speakers'])}


def is_speaker(name):
    return '扬声器' in name or 'Speakers' in name


def to_mp3(wav, mp3):
    subprocess.run(['/opt/homebrew/bin/lame', '--quiet', '-b', '128', str(wav), str(mp3)], check=True)


class Live:
    """The real-time engine and its recording. All settings except the gate preset change while running."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stream = None
        self.state = {'running': False, 'mode': None, 'error': '', 'note': ''}
        self.meter = {}
        self.set = {}

    # ---------------------------------------------------------------- live harmonizer
    def start_live(self, p):
        if self.stream is not None:
            raise RuntimeError('Already running: stop it first')
        try:
            return self._start_live(p)
        except Exception:
            self.stream = None
            self.state.update(running=False, mode=None)
            raise

    def _start_live(self, p):
        devs = sd.query_devices()
        i_in, i_out = int(p['in']), int(p['out'])
        out_name = devs[i_out]['name']
        spk = is_speaker(out_name)
        self.set = {'tonic': int(p.get('tonic', 0)), 'pre': float(p.get('pre_db', 12)), 'duet': float(p.get('duet_db', 0)),
                    'monitor': None if spk else float(p.get('monitor_db', -6)) if p.get('monitor', True) else None,
                    'gate': p.get('gate', 'normal')}
        fx.DEFAULTS.clear()
        fx.DEFAULTS.update(GATES.get(self.set['gate'], GATES['normal']))
        eng = fx.Harmonizer(tonic=self.set['tonic'], sr=SR, block=B, aec=spk)
        self.state.update(running=True, mode='live', error='', note='',
                          started=time.time(), devices=[devs[i_in]['name'], out_name], aec=spk)
        if spk:
            self.state['note'] = 'Sound check: 2 s of soft noise from the speaker, please stay quiet'
            rng = np.random.default_rng(0)
            ref = (10 ** (-35 / 20) * rng.standard_normal(2 * SR)).astype('float32')
            mic = sd.playrec(ref, samplerate=SR, channels=1, blocksize=B, latency=0.005, device=(i_in, i_out))[:, 0]
            sd.wait()
            erle = eng.calibrate(mic.astype(float), ref.astype(float))
            self.state['note'] = f'Sound check done (echo cancelled by {erle:.1f} dB)' if erle is not None else ''
        self.eng = eng
        self.rec_in, self.rec_out = [], []
        self.meter = {'in_db': -120.0, 'out_db': -120.0, 'gate': False, 'her': '', 'duet': '', 'xruns': 0,
                      'compute_ms': 0.0, 'peak_compute_ms': 0.0, 'gate_share': 0.0}
        self._gate_n = [0, 0]
        st = self.set

        def cb(indata, outdata, frames, t, status):
            t0 = time.perf_counter()
            if status:
                self.meter['xruns'] += 1
            raw = indata[:, 0].astype(float)
            xin = np.clip(raw * 10 ** (st['pre'] / 20), -1, 1)
            e = self.eng
            if e.tonic != st['tonic']:
                e.tonic = st['tonic']
            y = e.process(xin) * 10 ** (st['duet'] / 20)
            mon = 0.0 if st['monitor'] is None else 10 ** (st['monitor'] / 20)
            outdata[:, 0] = np.clip(y + mon * xin, -1, 1)
            if len(self.rec_in) < 10 * 60 * SR // B:                   # keep at most 10 minutes
                self.rec_in.append(raw.astype(np.float32))
                self.rec_out.append(y.astype(np.float32))
            m = self.meter
            m['in_db'] = 0.8 * m['in_db'] + 0.2 * db(xin)
            m['out_db'] = 0.8 * m['out_db'] + 0.2 * db(y)
            m['gate'] = bool(e.gate_open)
            self._gate_n[0] += 1
            self._gate_n[1] += int(e.gate_open)
            m['gate_share'] = self._gate_n[1] / self._gate_n[0]
            m['her'] = note_name(fx.hz_to_midi(e.f0)) if np.isfinite(e.f0) else ''
            m['duet'] = note_name(fx.third_below(e.note, e.tonic)) if (e.gate_open and e.note is not None) else ''
            dt = (time.perf_counter() - t0) * 1000
            m['compute_ms'] = 0.95 * m['compute_ms'] + 0.05 * dt
            m['peak_compute_ms'] = max(m['peak_compute_ms'], dt)

        self.stream = sd.Stream(samplerate=SR, blocksize=B, channels=1, latency=0.005, dtype='float32',
                                callback=cb, device=(i_in, i_out))
        self.stream.start()
        self.state['latency_ms'] = [round(1000 * v, 1) for v in self.stream.latency]
        return self.status()

    # ---------------------------------------------------------------- the original Duet: click + its own voice
    def start_duet(self, p):
        if self.stream is not None:
            raise RuntimeError('Already running: stop it first')
        try:
            return self._start_duet(p)
        except Exception:
            self.stream = None
            self.state.update(running=False, mode=None)
            raise

    def _start_duet(self, p):
        from dataclasses import replace
        D = duet_beat()
        lm, SRd, HOP = D['live_mic'], D['live_mic'].SR, D['live_mic'].HOP
        test = p.get('source') == 'test'
        style = p.get('style', 'bach')
        if style not in D['models']:
            self.state.update(note='Loading the chord model for this style…')
            D['models'][style] = lm.load_model(style)
        rng = (41, 65) if p.get('range') == 'low' else tuple(lm.ALTO)
        base = dict(D['res']['tuning']['chosen']['main'])
        base['alto'] = rng
        song = D['songs'].twinkle() if test else None
        bpm = float(song['bpm']) if test else float(p.get('bpm', 80))
        tonic = 0 if test else int(p.get('tonic', 0))
        K = song['K'] if test else int(p.get('beats', 32))
        cfg = replace(D['planner'].Config(**base), bpm=bpm, mode='long', J=int(p.get('join', 4)))
        logs = []
        brain = p.get('brain', 'policy')
        if brain == 'policy':
            self.state.update(note='Loading the leaderboard policy…')
            eng = D['policy_live'].PolicyDuet(D['policy_live'].load_policy(), D['models'][style], cfg, tonic=tonic, M=4, K=K,
                                              count_in=4, monitor=False, log=logs.append)
        else:
            eng = lm.LiveDuet(D['models'][style], cfg, tonic=tonic, M=4, K=K, count_in=4, monitor=False,
                              repeat=True, log=logs.append)
        devs = sd.query_devices()
        i_in, i_out = int(p['in']), int(p['out'])
        self.set = {'tonic': tonic, 'pre': 0.0, 'duet': float(p.get('duet_db', 0)), 'monitor': None,
                    'mode': 'duet', 'source': 'test' if test else 'mic', 'bpm': bpm, 'beats': K, 'style': style,
                    'range': list(rng), 'join': cfg.J, 'brain': brain,
                    'hpsync': (not test) and bool(p.get('hpsync', True)), 'her_db': float(p.get('her_db', 12))}
        sync = (not test) and p.get('sync') == 'phone'
        from countin import CountIn
        sy = {'on': sync, 'clicks': [], 'done': not sync, 'ci': CountIn(lm.SR)}
        if sync:                       # wait for the phone's four count-in clicks; Duet stays silent, no own clicks
            eng.t0 = 1e9
            eng._click = lambda n: np.zeros(n)
            self.set['sync'] = 'phone'
        her = None
        if test:
            her = D['audio'].render_her(song, eng.t0, eng.T_b, 'vibrato').astype(np.float32)
            her = np.concatenate([her, np.zeros(2 * SRd, np.float32)])
        self.duet = {'eng': eng, 'song': song, 'her': her, 'logs': logs, 'sr': SRd}
        self.rec_in, self.rec_out, self.rec_her = [], [], []
        self.meter = {'in_db': -120.0, 'out_db': -120.0, 'gate': False, 'her': '', 'duet': '', 'xruns': 0,
                      'compute_ms': 0.0, 'peak_compute_ms': 0.0, 'gate_share': 0.0, 'beat': 0, 'K': K, 'chord': '', 'phase': 'count-in'}
        st, dd = self.set, self.duet
        theory = D['theory']

        def cb(indata, outdata, frames, t, status):
            t0 = time.perf_counter()
            if status:
                self.meter['xruns'] += 1
            n = eng.n
            if test:                       # the singer is in the software: the engine hears her Li late
                li, lo = int(round(eng.Li * SRd)), int(round(eng.Lo * SRd))
                a = max(0, n - li)
                x = her[a:a + frames] if n - li >= 0 else np.zeros(frames, np.float32)
                x = np.pad(x, (0, frames - len(x)))
                hb = her[n + lo:n + lo + frames]
                hb = np.pad(hb, (0, frames - len(hb)))
            else:
                x = indata[:, 0].copy()
                hb = None
                if not sy['done']:     # count-in detector (countin.py): four evenly spaced clicks -> tempo and beat 1
                    r = sy['ci'].feed(x, eng.n / SRd - eng.Li)
                    if r is not None:
                        eng.T_b, eng.t0 = float(r[0]), float(r[1])
                        sy['done'] = True
                        self.state['note'] = f'Locked to the phone count-in: {60 / eng.T_b:.0f} BPM, beat 1 at {eng.t0:.2f} s'
            y = eng.process(x.astype(float)) * 10 ** (st['duet'] / 20) if frames == HOP else np.zeros(frames)
            if hb is not None:
                o = y + hb
            elif st.get('hpsync'):
                o = y + np.asarray(x, float) * 10 ** (st['her_db'] / 20)
            else:
                o = y
            outdata[:, 0] = np.clip(o, -1, 1)
            if outdata.shape[1] > 1:
                outdata[:, 1] = outdata[:, 0]
            self.rec_in.append(np.asarray(x, np.float32))
            self.rec_out.append(np.asarray(y, np.float32))
            m = self.meter
            m['in_db'] = 0.8 * m['in_db'] + 0.2 * db(x)
            m['out_db'] = 0.8 * m['out_db'] + 0.2 * db(y)
            th = eng.heard()
            k = int(np.floor((th - eng.t0) / eng.T_b)) if th >= 0 else -99
            m['phase'] = ('waiting for count-in…' if not sy['done'] else
                          (f'count-in {max(1, -k)}' if k < 0 else ('listening' if not eng.joined else 'singing')) if k < K else 'done')
            m['beat'] = max(0, min(k + 1, K))
            if getattr(eng, 'guess', None) is not None:
                m['chord'] = 'she next: ' + NAMES[(tonic + eng.guess) % 12]
            else:
                m['chord'] = theory.chord_name(eng.beats[k]['z_hat'], tonic) if 0 <= k < len(eng.beats) and eng.beats[k]['z_hat'] >= 0 else ''
            m['her'] = note_name(eng.ft_m[-1]) if eng.ft_v and eng.ft_v[-1] else ''
            m['duet'] = note_name(eng.last_note[2]) if eng.last_note and eng.last_note[2] is not None and eng.last_note[2] >= 0 else ''
            m['gate'] = bool(m['duet'])
            dt = (time.perf_counter() - t0) * 1000
            m['compute_ms'] = 0.95 * m['compute_ms'] + 0.05 * dt
            m['peak_compute_ms'] = max(m['peak_compute_ms'], dt)

        self.stream = sd.Stream(samplerate=SRd, blocksize=HOP, channels=(1, min(2, devs[i_out]['max_output_channels'])),
                                dtype='float32', latency='low', callback=cb, device=(i_in, i_out))
        eng.Li, eng.Lo = self.stream.latency
        if self.set['hpsync']:         # her (from the mic) and Duet both go out to the headphones on one timeline
            eng.Li, eng.Lo = 0.0, 0.0
        self.state.update(running=True, mode='duet', error='', started=time.time(), devices=[devs[i_in]['name'], devs[i_out]['name']],
                          aec=False, latency_ms=[round(1000 * v, 1) for v in self.stream.latency],
                          note=('Built-in singer: synthetic Twinkle, C major, 100 BPM' if test else ('Play the clip with the count-in on the phone' if sync else f'Sing with the click in your headphones: 4 clicks first, {bpm:.0f} BPM'))
                          )
        self.stream.start()

        def watch():
            while self.stream is not None and self.state.get('mode') == 'duet':
                if eng.done():
                    try:
                        self.stop()
                    except Exception as e:  # noqa: BLE001
                        self.state['error'] = str(e)
                    return
                time.sleep(0.2)
        threading.Thread(target=watch, daemon=True).start()
        return self.status()

    def update(self, p):
        if 'her_db' in p and self.set.get('mode') == 'duet':
            self.set['her_db'] = float(p['her_db'])
        if 'tonic' in p and self.set.get('mode') != 'duet':
            self.set['tonic'] = int(p['tonic'])
        if 'pre_db' in p:
            self.set['pre'] = float(p['pre_db'])
        if 'duet_db' in p:
            self.set['duet'] = float(p['duet_db'])
        if 'monitor_db' in p or 'monitor' in p:
            if self.state.get('aec'):
                self.set['monitor'] = None                                   # never on the speaker: feedback
            else:
                on = p.get('monitor', self.set['monitor'] is not None)
                self.set['monitor'] = float(p.get('monitor_db', self.set['monitor'] or -6)) if on else None
        return self.status()

    # ---------------------------------------------------------------- record one phrase (static modes)
    def start_rec(self, p):
        if self.stream is not None:
            raise RuntimeError('Already running: stop it first')
        try:
            return self._start_rec(p)
        except Exception:
            self.stream = None
            self.state.update(running=False, mode=None)
            raise

    def _start_rec(self, p):
        i_in = int(p['in'])
        self.rec_in, self.rec_out = [], []
        self.meter = {'in_db': -120.0, 'out_db': -120.0, 'gate': False, 'her': '', 'duet': '', 'xruns': 0,
                      'compute_ms': 0.0, 'peak_compute_ms': 0.0, 'gate_share': 0.0}
        maxb = 30 * SR // B

        def cb(indata, frames, t, status):
            if status:
                self.meter['xruns'] += 1
            x = indata[:, 0].astype(np.float32)
            if len(self.rec_in) < maxb:
                self.rec_in.append(x.copy())
            self.meter['in_db'] = 0.8 * self.meter['in_db'] + 0.2 * db(x)

        self.stream = sd.InputStream(samplerate=SR, blocksize=B, channels=1, dtype='float32', callback=cb, device=i_in)
        self.stream.start()
        self.state.update(running=True, mode='record', error='', note='Recording a phrase (max 30 s)', started=time.time(),
                          devices=[sd.query_devices(i_in)['name'], ''])
        return self.status()

    # ---------------------------------------------------------------- stop: write the recording
    def stop(self):
        with self.lock:
            return self._stop()

    def _stop(self):
        if self.stream is None:
            return self.status()
        self.stream.stop()
        self.stream.close()
        self.stream = None
        mode = self.state.get('mode')
        stamp = time.strftime('%m%d_%H%M%S')
        d = SESS / f'{mode}_{stamp}'
        d.mkdir(parents=True, exist_ok=True)
        x = np.concatenate(self.rec_in) if self.rec_in else np.zeros(1, np.float32)
        info = {'mode': mode, 'seconds': round(len(x) / SR, 2), 'devices': self.state.get('devices'),
                'settings': self.set if mode == 'live' else {}, 'meter_end': self.meter,
                'latency_ms': self.state.get('latency_ms')}
        if mode == 'duet':
            dd, D = self.duet, duet_beat()
            eng, SRd = dd['eng'], dd['sr']
            y = np.concatenate(self.rec_out) if self.rec_out else np.zeros(1, np.float32)
            li, lo = int(round(eng.Li * SRd)), int(round(eng.Lo * SRd))
            her_ac = dd['her'] if dd['her'] is not None else x[li:]          # Li = Lo = 0 in headphone-sync mode
            y_ac = np.concatenate([np.zeros(lo, np.float32), y])
            L = min(len(her_ac), len(y_ac))
            mix = np.clip(0.8 * her_ac[:L] / (np.abs(her_ac[:L]).max() + 1e-9) * 0.7 + y_ac[:L], -1, 1)
            sf.write(d / 'listen.wav', mix, SRd)
            to_mp3(d / 'listen.wav', d / 'listen.mp3')
            sf.write(d / 'mic_click_duet.wav', np.stack([x[:len(y)], y[:len(x)]], 1), SRd)
            summ = eng.summary()
            json.dump(summ, open(d / 'duet_summary.json', 'w'), default=float)
            sung = [b for b in summ['beats'] if b['h'] >= 0]
            info['duet'] = {'beats_decided': len(summ['beats']), 'beats_sung': len(sung), 'late_decisions': summ['late_decisions']}
            if dd['song'] is not None:                         # test singer: score against Twinkle's true notes
                from planner import Beat
                song = dd['song']
                bts = [Beat(k=b['k'], pos=b['pos'], z_true=song['beats'][b['k']]['z'], z_hat=b['z_hat'], q=b['q'],
                            action=b['action'], h=b['h']) for b in summ['beats']]
                smm = D['metrics'].summarize([D['metrics'].song_counts(song, bts, eng.cfg.J)])
                info['duet'].update({'consonant': round(float(smm['m3']), 3), 'chord_right': round(float(smm['m1']), 3)})
            info['files'] = {'listen': f'{d.name}/listen.mp3'}
            info['settings'] = self.set
        elif mode == 'live':
            y = np.concatenate(self.rec_out)
            sf.write(d / 'mic_duet.wav', np.stack([x, y], 1), SR)
            pre = 10 ** (self.set['pre'] / 20)
            mix = np.clip(0.7 * np.clip(x * pre, -1, 1) + 0.8 * y, -1, 1)
            sf.write(d / 'listen.wav', mix, SR)
            to_mp3(d / 'listen.wav', d / 'listen.mp3')
            info['files'] = {'listen': f'{d.name}/listen.mp3'}
        else:
            sf.write(d / 'phrase.wav', x, SR)
            to_mp3(d / 'phrase.wav', d / 'phrase.mp3')
            info['files'] = {'phrase': f'{d.name}/phrase.mp3'}
        json.dump(info, open(d / 'info.json', 'w'), indent=1, ensure_ascii=False, default=str)
        self.last_saved = d.name
        self.state.update(running=False, mode=None, note=f'Saved {d.name}')
        return self.status()

    def status(self):
        s = dict(self.state)
        s['elapsed'] = round(time.time() - s['started'], 1) if s.get('running') and s.get('started') else 0
        s['settings'] = self.set
        return s


LIVE = Live()
JOBS = {}


def sessions():
    out = []
    for d in sorted(SESS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        if not d.is_dir() or not (d / 'info.json').exists():
            continue
        info = json.load(open(d / 'info.json'))
        res = []
        for f in sorted(d.glob('*.mp3')):
            res.append(f'{d.name}/{f.name}')
        out.append({'name': d.name, 'mode': info.get('mode'), 'seconds': info.get('seconds'), 'mp3': res,
                    'settings': info.get('settings', {}), 'duet': info.get('duet')})
    return out


def process(name, kind, tonic=None):
    """Static modes on a recorded phrase. Runs the deliverables' own code in a subprocess."""
    d = SESS / name
    wav = d / 'phrase.wav'
    key = f'{name}:{kind}'
    JOBS[key] = {'state': 'running', 'started': time.time()}
    try:
        env = dict(os.environ, PYTHONWARNINGS='ignore')
        if kind in ('satb', 'styles'):
            if not (d / 'd1' / 'timing.json').exists():
                cmd = [PY, str(DUET / 'futures' / 'd1_record_harmonize' / 'run.py'), str(wav), '-o', str(d / 'd1')]
                if tonic is not None:
                    cmd += ['--tonic', str(tonic)]
                subprocess.run(cmd, cwd=DUET, check=True, capture_output=True, env=env, timeout=300)
            if kind == 'satb':
                shutil.copyfile(d / 'd1' / 'choir.mp3', d / 'satb_choir.mp3')
            else:
                subprocess.run([PY, str(DUET / 'futures' / 'live_test' / 'd4_phone.py'), name, str(wav),
                                str(d / 'd1' / 'timing.json')], cwd=DUET, check=True, capture_output=True, env=env, timeout=300)
                src = DUET / 'futures' / 'live_test' / 'd4_out' / f'{name}_bach_pop_jazz.mp3'
                shutil.copyfile(src, d / 'styles_bach_pop_jazz.mp3')
        elif kind == 'choir':
            code = ("import sys; from pathlib import Path; import core, choir; from make_demos import mp3\n"
                    f"x = core.load_audio(Path({str(wav)!r})); r = choir.process(x, 'psola', 'human')\n"
                    f"mp3(Path({str(d / 'own_choir.mp3')!r}), r['mix'])\n")
            subprocess.run([PY, '-c', code], cwd=DUET / 'futures' / 'd2_self_choir' / 'fix', check=True,
                           capture_output=True, env=env, timeout=300)
        JOBS[key] = {'state': 'done', 'seconds': round(time.time() - JOBS[key]['started'], 1)}
    except subprocess.CalledProcessError as e:
        JOBS[key] = {'state': 'error', 'error': (e.stderr or b'').decode(errors='ignore')[-600:]}
    except Exception as e:  # noqa: BLE001
        JOBS[key] = {'state': 'error', 'error': repr(e)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype):
        if not path.exists():
            return self._send(404, {'error': 'not found'})
        data = path.read_bytes()
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):                       # media players (Safari) read in ranges
            a, _, b = rng[6:].partition('-')
            a = int(a) if a else 0
            b = min(int(b) if b else len(data) - 1, len(data) - 1)
            part = data[a:b + 1]
            self.send_response(206)
            self.send_header('Content-Type', ctype)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Range', f'bytes {a}-{b}/{len(data)}')
            self.send_header('Content-Length', str(len(part)))
            self.end_headers()
            self.wfile.write(part)
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        try:
            if p in ('/', '/studio'):
                return self._file(HERE / 'studio.html', 'text/html; charset=utf-8')
            if p in ('/live', '/index.html'):
                return self._file(HERE / 'index.html', 'text/html; charset=utf-8')
            if p == '/api/studio':
                st = LIVE.status()
                return self._send(200, {'effects': studio.EFFECTS, 'sources': studio.sources(),
                                        'recording': st.get('running') and st.get('mode') == 'record', 'elapsed': st.get('elapsed', 0),
                                        'busy': bool(st.get('running'))})
            if p == '/api/devices':
                devs = devices(refresh='refresh=1' in (u.query or ''))
                return self._send(200, {'devices': devs, 'default': pick_default(devs)})
            if p == '/api/status':
                return self._send(200, LIVE.status())
            if p == '/api/sessions':
                return self._send(200, {'sessions': sessions(), 'jobs': JOBS})
            if p == '/api/events':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                while True:
                    msg = {'meter': LIVE.meter, 'status': LIVE.status()}
                    self.wfile.write(f'data: {json.dumps(msg, ensure_ascii=False, default=str)}\n\n'.encode())
                    self.wfile.flush()
                    time.sleep(0.1)
            if p.startswith('/files/'):
                rel = p[len('/files/'):]
                f = (SESS / rel).resolve()
                if SESS.resolve() not in f.parents:
                    return self._send(403, {'error': 'forbidden'})
                return self._file(f, 'audio/mpeg')
            if p.startswith('/test/'):
                f = TEST_CLIPS.get(p[len('/test/'):])
                return self._file(f, 'audio/mpeg') if f else self._send(404, {'error': 'not found'})
            return self._send(404, {'error': 'not found'})
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        u = urlparse(self.path)
        p = u.path
        n = int(self.headers.get('Content-Length') or 0)
        if p == '/api/studio/upload':
            raw = self.rfile.read(n)
            fname = unquote(u.query.split('name=', 1)[1]) if 'name=' in u.query else 'upload'
            safe = ''.join(ch for ch in os.path.basename(fname) if ch.isalnum() or ch in '._- ')[:60] or 'upload'
            stamp = time.strftime('%m%d%H%M%S')
            d = SESS / f'upload_{stamp}'
            d.mkdir(parents=True, exist_ok=True)
            (d / safe).write_bytes(raw)
            wav = d / 'source.wav'
            r = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-v', 'error', '-y', '-i', str(d / safe), '-ac', '1', '-ar', '48000', str(wav)],
                               capture_output=True)
            if r.returncode != 0 or not wav.exists():
                return self._send(400, {'error': 'could not read that audio file'})
            sid = 'up' + stamp
            studio.add_source(sid, safe, 'Uploaded file.', wav)
            studio.start(sid, str(wav))
            return self._send(200, {'ok': True, 'id': sid})
        body = json.loads(self.rfile.read(n) or b'{}')
        try:
            LIVE.state['error'] = ''
            if p == '/api/start':
                return self._send(200, LIVE.start_live(body))
            if p == '/api/studio/run':
                m = next((x for x in studio.sources() if x['id'] == body['id']), None)
                if m is None:
                    return self._send(404, {'error': 'unknown source'})
                studio.start(m['id'], m['wav'], force=bool(body.get('force')))
                return self._send(200, {'ok': True})
            if p == '/api/studio/record':
                return self._send(200, LIVE.start_rec(body))
            if p == '/api/studio/stop':
                LIVE.stop()
                name = getattr(LIVE, 'last_saved', None)
                if name and (SESS / name / 'phrase.wav').exists():
                    sid = 'rec' + time.strftime('%m%d%H%M%S')
                    studio.add_source(sid, 'Recording ' + time.strftime('%H:%M'), 'Recorded from the mic in this panel.',
                                      SESS / name / 'phrase.wav')
                    studio.start(sid, str(SESS / name / 'phrase.wav'))
                    return self._send(200, {'ok': True, 'id': sid})
                return self._send(200, {'ok': False})
            if p == '/api/duet':
                return self._send(200, LIVE.start_duet(body))
            if p == '/api/record':
                return self._send(200, LIVE.start_rec(body))
            if p == '/api/stop':
                return self._send(200, LIVE.stop())
            if p == '/api/update':
                return self._send(200, LIVE.update(body))
            if p == '/api/process':
                threading.Thread(target=process, args=(body['name'], body['kind'], body.get('tonic')), daemon=True).start()
                return self._send(200, {'ok': True})
            return self._send(404, {'error': 'not found'})
        except Exception as e:  # noqa: BLE001
            LIVE.state.update(error=str(e))
            traceback.print_exc()
            return self._send(500, {'error': str(e)})


def selftest():
    """No audio device: run the live engine on the phone take through the same per-block code path."""
    x, sr = sf.read(DUET / 'futures' / 'live_test' / 'phone_take_d1_song.wav')
    fx.DEFAULTS.clear()
    fx.DEFAULTS.update(GATES['normal'])
    eng = fx.Harmonizer(tonic=9, sr=SR, block=B, aec=False)
    notes, t0 = set(), time.perf_counter()
    for k in range(0, len(x) // B * B, B):
        y = eng.process(np.clip(x[k:k + B] * 10 ** (12 / 20), -1, 1))
        if eng.gate_open and eng.note is not None:
            notes.add((note_name(eng.note), note_name(fx.third_below(eng.note, eng.tonic))))
    print('selftest ok:', len(x) / SR, 's audio in', round(time.perf_counter() - t0, 2), 's; her->duet pairs seen:',
          sorted(notes)[:8])


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
        sys.exit(0)
    port = 8765
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    threading.Thread(target=studio.warm_builtins, daemon=True).start()     # pre-render the built-in clips
    print(f'Duet panel: http://127.0.0.1:{port}  (Ctrl-C to quit)', flush=True)
    srv.serve_forever()
