// The ear: microphone hops in, note on / note off out. Portable C++ (no Arduino), so the board, the Mac
// simulator and the unit tests run the same code.
//
// Lifted from the team's sketch: yinPitch() and the note decision of audioTask() are by Christine
// (GitHub cchristinem), file quantization_code_3, branch Christine of github.com/oldkingzz/music, commit bde720a
// (unchanged in quantization_code_4, branch SIgrid, commit ff33e2d, apart from one playback gate). The constants
// and the arithmetic are theirs, line for line. Changes, all mechanical:
//   * the I2S read, the prints, the MIDI / web / SD / OLED output and the UI mutex are gone: the caller hands in
//     one hop and gets the note events back;
//   * yinPitch's static d[] and audioTask's locals are members, so two trackers do not share state;
//   * Arduino's constrain() macro is pt_constrain() below, with the same expression;
//   * their playback gate (`&& playState == PLAY_IDLE`) is removed: a live duet listens while it sings.
// Every constant keeps its value; the names carry a PT_ prefix so they cannot collide with other headers.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace duet {

// ---- their pitch settings (quantization_code_4 lines 62-90), unchanged
constexpr uint32_t PT_SAMPLE_RATE = 16000;
constexpr int PT_FRAME = 768;
constexpr int PT_HOP = 256;
constexpr float PT_F_MIN = 75.0f;
constexpr float PT_F_MAX = 1100.0f;
constexpr int PT_TAU_MIN = (int)(PT_SAMPLE_RATE / PT_F_MAX);  // 14
constexpr int PT_TAU_MAX = (int)(PT_SAMPLE_RATE / PT_F_MIN);  // 213
constexpr int PT_W = PT_FRAME - PT_TAU_MAX;                   // 555

constexpr float PT_YIN_THRESHOLD = 0.15f;
constexpr float PT_MIN_CONFIDENCE = 0.60f;
constexpr float PT_RMS_GATE = 0.004f;
constexpr float PT_HYSTERESIS = 0.65f;

constexpr int PT_SETTLE_FRAMES = 5;
constexpr float PT_SETTLE_SPREAD = 0.6f;  // semitones
constexpr int PT_RELEASE_FRAMES = 6;      // unvoiced hops before note-off

// A retune breaks the build of the board, the simulator and the tests: these are the team's values.
static_assert(PT_SAMPLE_RATE == 16000 && PT_FRAME == 768 && PT_HOP == 256, "the team's SAMPLE_RATE, FRAME, HOP");
static_assert(PT_F_MIN == 75.0f && PT_F_MAX == 1100.0f, "the team's F_MIN, F_MAX");
static_assert(PT_TAU_MIN == 14 && PT_TAU_MAX == 213 && PT_W == 555, "YIN lag range as in the team's sketch");
static_assert(PT_YIN_THRESHOLD == 0.15f && PT_MIN_CONFIDENCE == 0.60f, "the team's YIN_THRESHOLD, MIN_CONFIDENCE");
static_assert(PT_RMS_GATE == 0.004f && PT_HYSTERESIS == 0.65f, "the team's RMS_GATE, HYSTERESIS");
static_assert(PT_SETTLE_FRAMES == 5 && PT_SETTLE_SPREAD == 0.6f && PT_RELEASE_FRAMES == 6,
              "the team's SETTLE_FRAMES, SETTLE_SPREAD, RELEASE_FRAMES");

// Arduino's constrain(amt, low, high) is ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt))).
template <class T>
static inline T pt_constrain(T amt, T low, T high) {
  return amt < low ? low : (amt > high ? high : amt);
}

enum : uint8_t { PT_NOTE_ON = 1, PT_NOTE_OFF = 2 };

struct PitchEvent {
  uint8_t kind;  // PT_NOTE_ON / PT_NOTE_OFF
  uint8_t midi;
  uint8_t vel;   // their velocity for NOTE_ON (20..127), 0 for NOTE_OFF
};

class PitchTracker {
 public:
  static const int MAX_EVENTS = 2;  // one hop gives at most: the old note off, the new note on

  PitchTracker() { reset(); }
  // Back to the state their audioTask starts in (frame of zeros, DC filter 0, no note).
  void reset();

  // One hop of PT_HOP I2S words exactly as their readBytes() gives them (32-bit slots, 24-bit sample on top).
  // Writes 0..MAX_EVENTS events to ev (in order) and returns how many.
  int hop_raw(const int32_t* raw, PitchEvent* ev);
  // One hop of int16 samples: fed as the 32-bit word x << 16, so their scaling (raw >> 8) / 2^23 gives x / 32768
  // exactly. Used for the fake microphone.
  int hop_i16(const int16_t* x, PitchEvent* ev);

  // Their yinPitch() on PT_FRAME samples: frequency in Hz (0 if none) and confidence 0..1.
  float yin(const float* x, float& confidence);

  // What their UI struct gets after each hop.
  int active() const { return active_; }        // the note that is on, -1 = none
  float freq() const { return ui_freq_; }        // Hz of this hop, 0 if unvoiced
  float cents() const { return ui_cents_; }      // -50..50 against the active note, 0 if none or unvoiced
  float conf() const { return ui_conf_; }        // YIN confidence of this hop (0 below the RMS gate)
  float rms() const { return rms_; }             // RMS of the 768-sample frame after the DC filter
  bool voiced() const { return voiced_; }
  float midi_f() const { return midi_f_; }       // fractional MIDI note of this hop if voiced

  // The 24-bit sample of one I2S word scaled to int16 (rounded, saturated): what MIC_HOP carries.
  static int16_t raw_to_i16(int32_t raw);

 private:
  int process(PitchEvent* ev);  // frame_ holds the newest hop: their steps 1) and 2)

  float frame_[PT_FRAME];
  float d_[PT_TAU_MAX + 2];
  float dc_;
  int active_;
  int hist_count_;
  int unvoiced_run_;
  float pitch_hist_[PT_SETTLE_FRAMES];
  float rms_hist_[PT_SETTLE_FRAMES];
  float ui_freq_, ui_cents_, ui_conf_, rms_, midi_f_;
  bool voiced_;
};

}  // namespace duet
