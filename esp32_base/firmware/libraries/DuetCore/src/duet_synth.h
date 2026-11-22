// The voice: note on / note off in, int16 samples out. Portable C++ (no Arduino).
//
// Lifted from the team's sketch: the small additive wavetable synth (buildWave, voiceNoteOn, voiceNoteOff,
// voicesIdle, renderVoices and the per-sample volume / clip / int16 step of their playback loop) is by Xinrui Gao
// (GitHub XinruiGao), file quantization_code_4, branch SIgrid of github.com/oldkingzz/music, commit ff33e2d.
// Same wavetable, same envelope constants, same voice allocation, same arithmetic. Changes, all mechanical:
//   * the sample rate is a parameter (default their PLAY_RATE 22050); renderVoices' function-static constants and
//     the globals voices[], wave[], fastRelease are members, so two synths do not share state;
//   * the master volume is a parameter of render() (their PLAY_VOLUME 0.35 is the cap SY_PLAY_VOLUME);
//   * MIDI-file playback, SD and the I2S writes are gone (the caller writes the rendered block).
//
// Threads: a Synth is used by ONE thread (the one that renders). Note commands from other threads go through a
// SynthQueue (single producer, single consumer, lock-free) and are applied by the rendering thread.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <atomic>

namespace duet {

// ---- their playback settings (quantization_code_4 lines 99-110), unchanged
constexpr uint32_t SY_PLAY_RATE = 22050;
constexpr int SY_PLAY_BLOCK = 256;          // frames rendered per I2S write
constexpr float SY_PLAY_VOLUME = 0.35f;     // master volume cap
constexpr int SY_NUM_VOICES = 4;
constexpr int SY_WAVE_BITS = 11;
constexpr int SY_WAVE_N = 1 << SY_WAVE_BITS;
constexpr float SY_ATTACK_SEC = 0.005f;
constexpr float SY_DECAY_TAU_SEC = 1.2f;
constexpr float SY_DECAY_FLOOR = 0.25f;
constexpr float SY_RELEASE_TAU_SEC = 0.06f;
constexpr float SY_FAST_RELEASE_SEC = 0.008f;  // the literal in their REL_FAST
constexpr int SY_STOP_FADE_BLOCKS = 30;        // their stop fade-out: at most 30 blocks
// A retune breaks the build of the board, the simulator and the tests: these are the team's values.
static_assert(SY_PLAY_RATE == 22050 && SY_PLAY_BLOCK == 256 && SY_PLAY_VOLUME == 0.35f && SY_NUM_VOICES == 4,
              "the team's PLAY_RATE, PLAY_BLOCK, PLAY_VOLUME, NUM_VOICES");
static_assert(SY_WAVE_BITS == 11 && SY_WAVE_N == 2048, "the team's WAVE_BITS, WAVE_N");
static_assert(SY_ATTACK_SEC == 0.005f && SY_DECAY_TAU_SEC == 1.2f && SY_DECAY_FLOOR == 0.25f &&
                  SY_RELEASE_TAU_SEC == 0.06f && SY_FAST_RELEASE_SEC == 0.008f,
              "the team's envelope constants");

class Synth {
 public:
  explicit Synth(uint32_t rate = SY_PLAY_RATE);
  uint32_t rate() const { return rate_; }

  void note_on(uint8_t note, uint8_t vel);  // their voiceNoteOn
  void note_off(uint8_t note);              // their voiceNoteOff
  bool idle() const;                        // their voicesIdle
  // All voices silent at once, fastRelease off (what their playback does before it starts).
  void reset();
  // Every gate off (normal release tails).
  void release_all();
  // Their stop: every gate off and the fast release on; render until idle() (at most SY_STOP_FADE_BLOCKS blocks).
  void release_all_fast();

  float render_sample();  // their renderVoices(): one mono sample, before the master volume
  // frames samples, each their `s = renderVoices() * volume; clip to +-1; (int16_t)(s * 32767)`. stereo: the same
  // value in L and R (2 * frames int16), else mono (frames int16). gain = master volume, 0..SY_PLAY_VOLUME.
  void render(int16_t* out, size_t frames, bool stereo, float gain);

  const float* wave() const { return wave_; }

 private:
  struct Voice {
    bool active, gate;
    uint8_t note;
    uint32_t phase, inc;
    float amp, attack, decay, rel;
  };
  void build_wave();  // their buildWave

  uint32_t rate_;
  float attack_step_, decay_factor_, rel_normal_, rel_fast_;
  Voice voices_[SY_NUM_VOICES];
  float wave_[SY_WAVE_N + 1];
  bool fast_release_;
};

// Volume byte of LIVE_CTRL (0..100, larger = 100) -> gain for Synth::render.
float synth_gain(uint8_t volume_percent);

enum : uint8_t { SC_NOTE_ON = 1, SC_NOTE_OFF = 2, SC_ALL_OFF = 3 };
struct SynthCmd {
  uint8_t op;  // SC_*
  uint8_t midi;
  uint8_t vel;
};
// Apply one command to a synth (rendering thread). SC_ALL_OFF releases every gate (normal release).
void synth_apply(Synth& s, const SynthCmd& c);

// Lock-free ring for SynthCmd: exactly one producer thread and one consumer thread (or callers serialised by a
// lock on either side). push() fails (returns false) when full; nothing blocks.
template <size_t N>
class SynthQueue {
  static_assert(N >= 2 && (N & (N - 1)) == 0, "N must be a power of two");

 public:
  SynthQueue() : head_(0), tail_(0), dropped_(0) {}
  bool push(const SynthCmd& c) {  // producer
    const uint32_t h = head_.load(std::memory_order_relaxed);
    if (h - tail_.load(std::memory_order_acquire) >= N) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    buf_[h & (N - 1)] = c;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }
  bool pop(SynthCmd* c) {  // consumer
    const uint32_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return false;
    *c = buf_[t & (N - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }
  uint32_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> head_, tail_, dropped_;
  SynthCmd buf_[N];
};

}  // namespace duet
