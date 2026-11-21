#!/bin/sh
# Compile one sketch for the ESP32-S3 N16R8 board. Compiles only: nothing is sent to the board.
#   sh tools/build_firmware.sh duet_base            (or duet_base_audio)
#   sh tools/build_firmware.sh duet_base_audio i2s  the real-audio variant (DUET_AUDIO_STANDIN=0), in its own folder
# Result: ~/work/tools/arduino/build/esp32_base_<sketch>/<sketch>.ino.bin  (the folder is about 40 MB);
#         the i2s variant goes to ~/work/tools/arduino/build/esp32_base_<sketch>_i2s/.
# duet_base_audio without a variant is the stand-in build (DUET_AUDIO_STANDIN=1, the sketch's default).
# tools/upload_firmware.sh uploads esp32_base_<sketch> only, i.e. not the i2s folder.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SKETCH="${1:-duet_base}"
VARIANT="${2:-}"
. "$HERE/tools/board.env"
case "$VARIANT" in
  "")
    BUILD="$BUILD_ROOT/esp32_base_$SKETCH"
    arduino-cli --config-file "$ARDUINO_CONFIG" compile -b "$FQBN" \
      --libraries "$HERE/firmware/libraries" --warnings all \
      --build-path "$BUILD" "$HERE/firmware/$SKETCH"
    ;;
  i2s)
    [ "$SKETCH" = "duet_base_audio" ] || { echo "variant i2s exists for duet_base_audio only"; exit 1; }
    BUILD="$BUILD_ROOT/esp32_base_${SKETCH}_i2s"
    arduino-cli --config-file "$ARDUINO_CONFIG" compile -b "$FQBN" \
      --libraries "$HERE/firmware/libraries" --warnings all \
      --build-property "compiler.cpp.extra_flags=-DDUET_AUDIO_STANDIN=0" \
      --build-path "$BUILD" "$HERE/firmware/$SKETCH"
    ;;
  *)
    echo "unknown variant '$VARIANT' (known: i2s)"
    exit 1
    ;;
esac
ls -l "$BUILD/$SKETCH.ino.bin"
