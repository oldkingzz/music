// Stand-in audio: a made-up microphone and a counting speaker, for a board with no audio hardware wired.
// Portable C++, integer arithmetic only (no sin, no floats), so the board, the Mac simulator, the unit tests and
// the Python mirror (host/duet_host/fake_audio.py) produce exactly the same numbers.
//
// Microphone: "Twinkle, Twinkle" in C major, 14 notes (C4 C4 G4 G4 A4 A4 G4- F4 F4 E4 E4 D4 D4 C4-) on a 0.5 s
// grid, the two notes marked - last 1.0 s; the loop is 8.0 s = 192000 samples at 24 kHz. Every note sounds for
// its slot minus a 50 ms silent gap (so note starts can be found), with a 5 ms linear attack and a 10 ms linear
// release. Tone: parabolic sine of a 32-bit phase, fundamental + 2nd + 3rd harmonic with weights 6 : 3 : 2,
// peak 11999. The sample is a pure function of its index n: standin_mic_sample(n).
//
// Speaker: takes 240-sample frames and keeps frames played, silent frames, the peak |sample| and a running
// CRC-32 (zlib / IEEE 802.3, the same as Python's zlib.crc32(bytes, prev)) of the little-endian bytes of every
// sample of every audio frame. Silent frames (jitter buffer priming / underrun) are counted, not summed, so the
// host can predict the checksum from what it sent.
//
// Names here avoid the macros of firmware/duet_base_audio/config.h (FRAME_SAMPLES, AUDIO_FS, ...).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <atomic>

namespace duet {

static const uint32_t STANDIN_RATE = 24000;
static const size_t STANDIN_FRAME = 240;  // samples per 10 ms frame

struct StandinNote {
  uint8_t midi;
  uint32_t inc;   // phase increment per sample = round(f * 2^32 / 24000)
  uint32_t slot;  // samples from this note's start to the next one's (sound + gap)
};

static const size_t STANDIN_NOTES = 14;
static const uint32_t STANDIN_GAP = 1200;      // 50 ms of silence at the end of every slot
static const uint32_t STANDIN_ATTACK = 120;    // 5 ms
static const uint32_t STANDIN_RELEASE = 240;   // 10 ms
static const uint32_t STANDIN_LOOP = 192000;   // 8.0 s
static const int32_t STANDIN_W1 = 6, STANDIN_W2 = 3, STANDIN_W3 = 2;
static const int64_t STANDIN_GAIN_MUL = 5333;  // out = mix * env * GAIN_MUL >> GAIN_SHIFT
static const int STANDIN_GAIN_SHIFT = 31;

extern const StandinNote STANDIN_MELODY[STANDIN_NOTES];

// Parabolic sine of a 32-bit phase: -16384..16384.
int32_t standin_psin(uint32_t phase);
// Microphone sample n (n = 0 is the first sample after the mic gate opened).
int16_t standin_mic_sample(uint32_t n);
// count samples starting at n0.
void standin_mic_fill(uint32_t n0, int16_t* out, size_t count);

// Continue a CRC-32 over int16 samples (little-endian bytes). crc = 0 to start; same as zlib.crc32(b, crc).
uint32_t crc32_samples(uint32_t crc, const int16_t* x, size_t n);

class StandinSpeaker {
 public:
  StandinSpeaker() { reset(); }
  void reset();
  // An audio frame of STANDIN_FRAME samples reached the "speaker".
  void play(const int16_t* frame);
  // A silent frame (nothing to play: jitter buffer priming or underrun).
  void silence() { silent.fetch_add(1, std::memory_order_relaxed); }

  // Written by the one task that plays; read from any task.
  std::atomic<uint32_t> frames;  // audio frames played
  std::atomic<uint32_t> silent;  // silent frames
  std::atomic<uint32_t> crc;     // CRC-32 of every sample of every audio frame, in order
  std::atomic<uint32_t> peak;    // max |sample| of the audio frames
};

}  // namespace duet
