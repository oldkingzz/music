#!/bin/sh
# Send an already compiled sketch to the board. This REPLACES the program on the board.
#   sh tools/upload_firmware.sh duet_base --yes-upload
#   sh tools/upload_firmware.sh duet_base_audio --yes-upload
# Without the literal --yes-upload it only says what it would do and exits 1 (it touches nothing).
# Compile first with  sh tools/build_firmware.sh <sketch>.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH="$1"
FLAG="$2"
. "$HERE/tools/board.env"
BUILD="$BUILD_ROOT/esp32_base_$SKETCH"
PY="${DUET_PY:-$HERE/../duet_v0/.venv/bin/python}"   # set DUET_PY to use another Python

# 1. Refuse unless asked explicitly. Nothing below this block runs without the flag.
if [ -z "$SKETCH" ] || [ "$FLAG" != "--yes-upload" ]; then
  echo "Not uploading. This would:"
  echo "  - find the board by USB id $USB_VID:$USB_PID (exactly one must be plugged in)"
  echo "  - write $BUILD/${SKETCH:-<sketch>}.ino.bin to it, replacing whatever program is on the board"
  echo "To do it:  sh tools/upload_firmware.sh ${SKETCH:-<sketch>} --yes-upload"
  exit 1
fi

[ -f "$BUILD/$SKETCH.ino.bin" ] || { echo "No build at $BUILD. Run: sh tools/build_firmware.sh $SKETCH"; exit 1; }

# 2. Find the board by USB id, never by port name (the name changes between plugs).
PORTS="$("$PY" -c "from serial.tools import list_ports
print(' '.join(p.device for p in list_ports.comports() if p.vid == 0x$USB_VID and p.pid == 0x$USB_PID))")"
COUNT=$(echo $PORTS | wc -w | tr -d ' ')
if [ "$COUNT" -eq 0 ]; then
  echo "No board with USB id $USB_VID:$USB_PID found. Is the cable in the board's USB (not UART) port?"
  echo "First upload (the factory program is still on the board, it shows up as 303A:4001):"
  echo "hold BOOT, tap RESET, release BOOT, then run this again."
  exit 1
fi
if [ "$COUNT" -gt 1 ]; then
  echo "More than one board found ($PORTS). Unplug all but one and run again."
  exit 1
fi
PORT="$PORTS"

# 3. Upload.
echo "Uploading $SKETCH to $PORT. This replaces whatever program is on the board."
echo "If the board does not respond: hold BOOT, tap RESET, release BOOT, then run this again."
arduino-cli --config-file "$ARDUINO_CONFIG" upload -p "$PORT" -b "$FQBN" --input-dir "$BUILD"
