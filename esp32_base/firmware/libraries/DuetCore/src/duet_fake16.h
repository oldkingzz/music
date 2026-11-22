// The fake microphone of the live chain: a known melody at 16 kHz, the pitch tracker's rate. Portable C++,
// integer arithmetic only, so the board, the Mac simulator, the unit tests and the Python mirror
// (host/duet_host/fake_audio.py, the *16 functions) produce exactly the same numbers.
//
// The same method as the 24 kHz stand-in of duet_standin.h (parabolic sine of a 32-bit phase, harmonics
// 6 : 3 : 2, out = |mix| * env * 5333 >> 31, 5 ms linear attack, 10 ms linear release), with its own tables:
// that stand-in is baked for 24 kHz and would come out a fifth low and 1.5 x too long at 16 kHz.
//
// Melody: "Twinkle, Twinkle" in C major, 14 notes (MIDI 60 60 67 67 69 69 67 65 65 64 64 62 62 60) on a 0.5 s
// grid, notes 7 and 14 last 1.0 s; loop 8.0 s = 128000 samples. Every note sounds for its slot minus a silent gap.
// The gap is FAKE16_GAP (see there for why it is that long). fake16_sample(n) is a pure function of n.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace duet {

static const uint32_t FAKE16_RATE = 16000;
static const size_t FAKE16_NOTES = 14;
static const uint32_t FAKE16_SHORT = 8000;    // 0.5 s slot
static const uint32_t FAKE16_LONG = 16000;    // 1.0 s slot
static const uint32_t FAKE16_LOOP = 128000;   // 8.0 s
static const uint32_t FAKE16_ATTACK = 80;     // 5 ms
static const uint32_t FAKE16_RELEASE = 160;   // 10 ms
// Silent gap at the end of every slot: 160 ms. The team's tracker releases a note only after 6 unvoiced hops
// (96 ms) and its analysis frame is 48 ms long, so the 50 ms gap of the 24 kHz stand-in makes repeated notes
// (60 60, 67 67, ...) merge into one. 160 ms gives 14 separate notes (pinned in host_tests/test_live.cpp).
static const uint32_t FAKE16_GAP = 2560;
static const uint32_t FAKE16_GAP_50MS = 800;  // the old stand-in's gap, kept for the test that pins the merge

struct Fake16Note {
  uint8_t midi;
  uint32_t inc;   // phase increment per sample = round(f * 2^32 / 16000), f = 440 * 2^((midi - 69) / 12)
  uint32_t slot;  // samples from this note's start to the next one's
};
extern const Fake16Note FAKE16_MELODY[FAKE16_NOTES];

// Sample n of the fake microphone (n = 0: the first sample after LISTEN went on). gap: silent samples at the end
// of each slot (FAKE16_GAP unless a test asks for another one; gap < 8000 - attack - release).
int16_t fake16_sample(uint32_t n, uint32_t gap = FAKE16_GAP);
void fake16_fill(uint32_t n0, int16_t* out, size_t count, uint32_t gap = FAKE16_GAP);

}  // namespace duet
