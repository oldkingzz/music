"""Duet live test page, version 11 = page v10 plus a sound-device watchdog.

Vincent's 11:29 run was silent: the AirPods Max dropped off the Mac during the run (PortAudio listed no microphone and
no AirPods output right after; Bluetooth showed it connected again a minute later; AirPods switch to an iPhone / iPad
that starts playing). The stream stopped calling back, nothing was heard, and the page just waited (33.8 s, a 2-block
recording). v11 records when the audio callback last ran; if a run or a calibration gets no callback for 1.5 s, it ends
it and says that the headphones / microphone stopped answering. Everything else is page v10. Serves live_panel_v10.html.

  cd duet_v0 && .venv/bin/python futures/v4_selfiter/live_panel_v11.py        # then open http://127.0.0.1:8766
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_panel_v10 as P10  # noqa: E402

P3 = P10.P3
STALL_S = 1.5
STALL_MSG = ('声音设备没有反应了：耳机或麦克风断开了（AirPods 可能自动切到了正在放声音的手机或 iPad）。'
             '在 Mac 上重新连好 AirPods，刷新页面再试。')
_cb_prev, _cb_file_prev = P3.Session._cb, P3.Session._cb_file
_start_prev, _calibrate_prev = P3.Session.start, P3.Session.calibrate


def _cb(self, indata, outdata, frames, t, status):
    self.last_cb = time.time()
    return _cb_prev(self, indata, outdata, frames, t, status)


def _cb_file(self, outdata, frames, t, status):
    self.last_cb = time.time()
    return _cb_file_prev(self, outdata, frames, t, status)


def watchdog(s, states=('running', 'calibrating'), stall=STALL_S):
    """End the current run / calibration when the audio callback has been silent for `stall` seconds."""
    while s.state in states and not s.done.is_set():
        time.sleep(0.2)
        if time.time() - getattr(s, 'last_cb', time.time()) > stall:
            s.msg = STALL_MSG
            s.done.set()
            return True
    return False


def _watched(fn):
    def run(self, p, source=None, sim=None):
        st = fn(self, p, source=source, sim=sim)
        if source is None:                              # a live stream (offline feeds call _cb themselves)
            self.last_cb = time.time() + 1.0            # grace while the stream starts
            threading.Thread(target=watchdog, args=(self,), daemon=True).start()
        return st
    return run


P3.Session._cb, P3.Session._cb_file = _cb, _cb_file
P3.Session.start, P3.Session.calibrate = _watched(_start_prev), _watched(_calibrate_prev)

if __name__ == '__main__':
    print('page v11 = page v10 + sound-device watchdog', flush=True)
    P3.main()
