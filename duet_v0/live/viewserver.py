"""Live view: a tiny local web server (stdlib http.server) for the demo video.

    GET /              live/view/index.html
    GET /state.json    current engine state (polled ~15x per second by the page)
    GET /reference.json  the piece: reference notes, harmony plan, breaths (fetched once)

Binds to 127.0.0.1 only. Runs in a daemon thread; reads engine state, never writes it.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEW_DIR = Path(__file__).resolve().parent / 'view'


class ViewServer:
    def __init__(self, engine, port=8765, host='127.0.0.1', extra=None):
        self.engine = engine
        self.extra = extra or {}
        self.requests = 0
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):            # keep the terminal quiet
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                srv.requests += 1
                path = self.path.split('?')[0]
                try:
                    if path in ('/', '/index.html'):
                        self._send(200, (VIEW_DIR / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                    elif path == '/state.json':
                        st = None
                        for _ in range(3):             # retry if the audio thread mutated mid-read
                            try:
                                st = srv.engine.view_state()
                                break
                            except RuntimeError:
                                continue
                        st = st or {'error': 'busy'}
                        st.update(srv.extra)
                        self._send(200, json.dumps(st).encode(), 'application/json')
                    elif path == '/reference.json':
                        self._send(200, json.dumps(srv.engine.reference_view()).encode(), 'application/json')
                    else:
                        self._send(404, b'not found', 'text/plain')
                except BrokenPipeError:
                    pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.2},
                                       daemon=True, name='view-http')

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
