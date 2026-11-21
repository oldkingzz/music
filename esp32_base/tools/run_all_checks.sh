#!/bin/sh
# Run every check in order and stop at the first failure. The audio check (last) plays and records for about
# 30 s; against a board it opens the speaker and mic gates (the stand-in build drives no audio pin).
#   sh tools/run_all_checks.sh          (sim: the Mac stand-in, no board needed)
#   sh tools/run_all_checks.sh auto     (the board, found by USB id: this OPENS its serial port)
#   sh tools/run_all_checks.sh /dev/cu.usbmodem101
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${DUET_PY:-$HERE/../duet_v0/.venv/bin/python}"   # set DUET_PY to use another Python
T="${1:-sim}"
cd "$HERE" || exit 1

step() {  # step <name> <command...>
  NAME="$1"; shift
  echo "== $NAME"
  "$@" || { echo "FAILED at step: $NAME"; exit 1; }
}

step "core unit tests + simulator build" sh firmware/host_tests/build_and_test.sh testdata/note_vectors_val_v1.txt
if [ -f host/tests/test_host.py ]; then
  step "host unit tests" "$PY" host/tests/test_host.py
else
  echo "== host unit tests: host/tests/test_host.py not found, skipped"
fi
step "audio unit tests" "$PY" host/tests/test_audio.py
step "link check ($T)" "$PY" host/checks/check_link.py --target "$T"
step "note check ($T)" "$PY" host/checks/check_notes.py --target "$T"
step "audio check ($T)" "$PY" host/checks/check_audio.py --target "$T"
echo "All checks passed (target $T)."
