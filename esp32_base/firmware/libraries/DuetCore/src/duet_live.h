// The live chain of sketch duet_live: everything between the I2S calls. Portable C++ (no Arduino), so the
// sketch and the Mac simulator (board_sim --live) run exactly this code.
//
//   hop (fake melody or real mic) -> PitchTracker -> HEARD_ON / HEARD_OFF
//        -> the configured plug-in (Engine::plugin_step) -> SUNG_ON / SUNG_OFF -> synth commands (LiveVoice::q)
//   and to the host, through Engine::send: LIVE_NOTE per event, LIVE_STATE every 6 hops, MIC_HOP per hop (monitor)
//
// Threads. LiveChain is not thread-safe by itself. On the board ONE lock serialises every call into the Engine
// and into LiveChain (feed / idle in the USB RX task, the hop calls in the microphone task, the I2S error LOG):
// CONFIG (inside Engine::feed) swaps the plug-in that the hop calls, and both send through the Engine's one
// transmit buffer. The gate getters (flags(), listening(), gain(), epoch() ...) are atomic and may be read by any
// task without the lock. The synth commands go to LiveVoice::q, a single-producer single-consumer ring: the
// producer side is always a caller holding that lock, the consumer is the speaker task.
//
// Timing: one hop = 256 samples at 16 kHz = 16 ms. t_ms of an event = (hops processed since LISTEN went on,
// including the hop that caused it) x 16. The plug-in gets onset = t_ms / 1000 * bpm / 60 beats and prev_dur =
// (HEARD_OFF t_ms - HEARD_ON t_ms of her previous note) / 1000 * bpm / 60 beats (negative for the first note).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "duet_engine.h"
#include "duet_fake16.h"
#include "duet_pitch.h"
#include "duet_protocol.h"
#include "duet_synth.h"

namespace duet {

static_assert(PT_HOP == MIC_HOP_SAMPLES, "one MIC_HOP carries one tracker hop");
static_assert(PT_SAMPLE_RATE == FAKE16_RATE, "the fake microphone runs at the tracker's rate");
static_assert(PT_HOP * 1000 / PT_SAMPLE_RATE == LIVE_HOP_MS && (PT_HOP * 1000) % PT_SAMPLE_RATE == 0,
              "one hop is a whole number of ms");

static const uint8_t LIVE_DEFAULT_BPM = 120;

// The speaker side: the synth and the command ring. Owned by the speaker task on the board (or the stand-in
// speaker sink of the simulator); only `q` and `blocks` are touched from elsewhere.
class LiveVoice {
 public:
  LiveVoice() : synth(SY_PLAY_RATE), blocks(0) {}
  // The speaker just came up: every voice silent (their playback start). Queued commands are kept.
  void start() { synth.reset(); }
  // Apply the queued commands, render one SY_PLAY_BLOCK block of 16-bit stereo (2 * 256 int16) at gain, count it.
  void block(int16_t* lr, float gain);
  // Their stop fade: call begin_stop() once, then stop_block() until it returns false (it renders and counts a
  // block only when it returns true): every gate off, fast release, until idle or SY_STOP_FADE_BLOCKS blocks.
  void begin_stop();
  bool stop_block(int16_t* lr, float gain);
  // The speaker is off: drop whatever was queued (nothing is rendered).
  void drop_pending();

  Synth synth;
  SynthQueue<64> q;
  std::atomic<uint32_t> blocks;  // synth blocks handed to the speaker since boot

 private:
  int stop_left_ = 0;
};

// Hardware facts the chain reports in LIVE_STATE; the platform fills them.
struct LiveHw {
  bool mic_up;   // microphone I2S running (simulator: its stand-in silence source)
  bool spk_up;   // speaker I2S running (simulator: the counting sink)
  bool error;    // an I2S begin failed
};
typedef void (*LiveHwFn)(LiveHw* hw, void* ctx);

class LiveChain {
 public:
  LiveChain();
  void begin(Engine* engine, LiveVoice* voice, LiveHwFn hw, void* hw_ctx);

  // A frame the engine passed through. Handles LIVE_CTRL and SYNTH_NOTE and returns true for them.
  bool on_frame(const Frame& f);
  // LIVE_CTRL. reply_seq >= 0: answer LIVE_STATE with that seq and LS_REPLY.
  void control(const LiveCtrl& c, int reply_seq);
  // SYNTH_NOTE: ignored unless SING is on.
  void synth_note(uint8_t midi, uint8_t vel);

  // One hop of the source. Call only while listening(): hop_fake() for the fake melody, hop_raw() with the
  // PT_HOP 32-bit I2S words of the real microphone.
  void hop_fake();
  void hop_raw(const int32_t* raw);

  // Gates in force (atomic, any task).
  uint8_t flags() const { return flags_.load(); }
  bool listening() const { return (flags_.load() & LC_LISTEN) != 0; }
  bool singing() const { return (flags_.load() & LC_SING) != 0; }
  bool fake() const { return (flags_.load() & LC_FAKE) != 0; }
  bool want_mic() const { return (flags_.load() & (LC_LISTEN | LC_FAKE)) == LC_LISTEN; }  // real mic needed
  uint8_t volume() const { return volume_.load(); }
  float gain() const { return synth_gain(volume_.load()); }
  // +1 at every restart of the source (rising LISTEN, FAKE change while listening): restart the hop clock.
  uint32_t epoch() const { return epoch_.load(); }

  uint32_t hops() const { return hops_; }
  int sung() const { return sung_; }
  const PitchTracker& tracker() const { return tracker_; }
  void send_state(uint8_t seq, bool reply);

  // Tests only: the fake melody's silent gap (FAKE16_GAP by default).
  void set_fake_gap(uint32_t gap) { fake_gap_ = gap; }

 private:
  void restart_source();
  void release_all(uint8_t old_flags);
  void after_hop(const int16_t* x16, int n_ev, const PitchEvent* ev);
  void end_sung(uint8_t src, uint32_t t_ms);
  void emit(uint8_t kind, uint8_t midi, uint8_t vel, uint8_t fl, uint32_t t_ms);
  void synth_cmd(uint8_t op, uint8_t midi, uint8_t vel);
  float beats(uint32_t ms) const { return (float)ms / 1000.0f * (float)bpm_ / 60.0f; }
  uint32_t now_ms() const { return listening() ? hops_ * LIVE_HOP_MS : 0; }

  Engine* engine_;
  LiveVoice* voice_;
  LiveHwFn hw_;
  void* hw_ctx_;
  PitchTracker tracker_;
  std::atomic<uint8_t> flags_;
  std::atomic<uint8_t> volume_;
  std::atomic<uint32_t> epoch_;
  uint8_t bpm_;
  uint32_t fake_gap_;
  uint32_t fake_pos_;  // next fake sample index
  uint32_t hops_;
  uint16_t heard_n_, sung_n_;
  uint8_t note_seq_, state_seq_, hop_seq_;
  int heard_i_;         // index of her next note for the plug-in
  uint32_t heard_on_ms_;
  bool have_prev_;
  uint32_t prev_dur_ms_;
  int sung_;            // Duet's note from the plug-in, -1 = none
  // latest hop
  float freq_;
  uint8_t conf_;
  int8_t level_, cents_;
  int16_t x16_[PT_HOP];
  uint8_t hop_payload_[MIC_HOP_LEN];
};

// ------------------------------------------------------------------ INFO of duet_live
// Platform facts for INFO. The engine puts fw, ver, proto, plugins in front (56 bytes for duet_live); the whole
// payload is capped at MAX_TEXT = 240, so live_info appends whole key=value fields only, in this order of need:
// flash, psram (first-contact check), the live keys (check_live), then the rest of sketch 1's keys; a field that
// does not fit is left out, never cut.
struct LiveInfoFacts {
  const char* chip;
  unsigned rev, cpu_mhz;
  uint32_t flash, psram, psram_free, heap_free;
  int rgb_gpio;
  uint32_t tx_dropped, uptime_ms;
  uint32_t mic_rate, spk_rate;
  int mic_pins[3];  // BCLK, WS, SD
  int spk_pins[3];  // BCLK, LRC, DIN
};
// Writes "flash=..;psram=..;live=1;mic_rate=..;spk_rate=..;mic_pins=a/b/c;spk_pins=a/b/c;chip=..;rev=..;
// cpu_mhz=..;heap_free=..;rgb_gpio=..;tx_dropped=..;psram_free=..;uptime_ms=.." (no trailing ';'), NUL-terminated,
// at most cap - 1 characters. Returns the length.
size_t live_info(char* buf, size_t cap, const LiveInfoFacts& f);
// Only the first part: flash, psram and the live keys (the simulator appends its own facts after it).
size_t live_info_head(char* buf, size_t cap, const LiveInfoFacts& f);
// Append text to buf[0..n) if all of it fits (with its NUL) in cap; returns the new length.
size_t info_append(char* buf, size_t cap, size_t n, const char* text);

}  // namespace duet
