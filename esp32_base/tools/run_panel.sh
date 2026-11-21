#!/bin/sh
# Start the local web panel:  sh tools/run_panel.sh [port]   (default 8765), then open http://localhost:<port>
# Starting it opens no serial port, no microphone and no browser; it only serves the page.
# The board is opened only when you ask for it in the panel. Stop with Ctrl-C.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${DUET_PY:-$HERE/../duet_v0/.venv/bin/python}"   # set DUET_PY to use another Python
cd "$HERE/host"
exec "$PY" -m duet_host.webapp --port "${1:-8765}"
