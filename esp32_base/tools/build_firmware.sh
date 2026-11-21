#!/bin/sh
# Compile one sketch for the ESP32-S3 N16R8 board. Compiles only: nothing is sent to the board.
#   sh tools/build_firmware.sh duet_base            (or duet_base_audio)
# Result: ~/work/tools/arduino/build/esp32_base_<sketch>/<sketch>.ino.bin  (the folder is about 40 MB).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH="${1:-duet_base}"
. "$HERE/tools/board.env"
arduino-cli --config-file "$ARDUINO_CONFIG" compile -b "$FQBN" \
  --libraries "$HERE/firmware/libraries" --warnings all \
  --build-path "$BUILD_ROOT/esp32_base_$SKETCH" "$HERE/firmware/$SKETCH"
ls -l "$BUILD_ROOT/esp32_base_$SKETCH/$SKETCH.ino.bin"
