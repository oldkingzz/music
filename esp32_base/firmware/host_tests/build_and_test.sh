#!/bin/sh
# Compile DuetCore for the Mac and run it: unit tests (with sanitizers) and the board simulator.
#   sh firmware/host_tests/build_and_test.sh [note_vectors.txt ...]
# Writes firmware/host_tests/build/{test_core,board_sim} (about 0.4 MB together).
# The speaker jitter buffer is the sketch's own file (firmware/duet_base_audio/jitter_buffer.cpp, with its
# config.h), compiled unchanged, so the simulator and the tests run the code the board runs.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../libraries/DuetCore/src"
AUD="$HERE/../duet_base_audio"
mkdir -p "$HERE/build"
CORE="$SRC/duet_protocol.cpp $SRC/duet_engine.cpp $SRC/duet_harmonizer.cpp $SRC/duet_standin.cpp"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -I"$SRC" -I"$AUD" \
  "$HERE/test_core.cpp" $CORE "$AUD/jitter_buffer.cpp" -o "$HERE/build/test_core"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$SRC" -I"$AUD" "$HERE/board_sim.cpp" $CORE "$AUD/jitter_buffer.cpp" \
  -o "$HERE/build/board_sim"
"$HERE/build/test_core" "$@"
