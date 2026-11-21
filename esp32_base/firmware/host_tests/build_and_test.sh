#!/bin/sh
# Compile DuetCore for the Mac and run it: unit tests (with sanitizers) and the board simulator.
#   sh firmware/host_tests/build_and_test.sh [note_vectors.txt ...]
# Writes firmware/host_tests/build/{test_core,board_sim} (about 0.3 MB together).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../libraries/DuetCore/src"
mkdir -p "$HERE/build"
CORE="$SRC/duet_protocol.cpp $SRC/duet_engine.cpp $SRC/duet_harmonizer.cpp"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined -I"$SRC" \
  "$HERE/test_core.cpp" $CORE -o "$HERE/build/test_core"
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$SRC" "$HERE/board_sim.cpp" $CORE -o "$HERE/build/board_sim"
"$HERE/build/test_core" "$@"
