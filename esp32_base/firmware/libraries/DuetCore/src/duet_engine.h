// The board's brain without the board: bytes from the host go in, frames to the host come out.
// Portable C++. The sketch gives it the serial port and the clock; the Mac simulator gives it stdin/stdout.
//
// The engine answers these itself:
//   PING     -> PONG        (same seq; echoes the host time, adds the device clock)
//   HELLO    -> INFO        (same seq; firmware name, version, plug-ins, then whatever the platform adds)
//   CONFIG   -> CONFIG_ACK  (same seq; selects a plug-in and calls its start())
//   NOTE_IN  -> NOTE_OUT    (same seq; calls the plug-in's step() and times it)
// Every other valid frame (audio, servo, LED, AUDIO_CTRL) is handed to the passthrough callback, if set.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "duet_harmonizer.h"
#include "duet_notes.h"
#include "duet_protocol.h"

namespace duet {

typedef void (*WriteFn)(const uint8_t* data, size_t n, void* ctx);  // one whole frame per call
typedef uint32_t (*MicrosFn)();
// Appends platform facts as "key=value;key=value" (no trailing ';'). Returns the number of bytes written.
typedef size_t (*InfoFn)(char* buf, size_t cap, void* ctx);

struct EngineStats {
  uint32_t pings;
  uint32_t hellos;
  uint32_t configs;
  uint32_t notes;
  uint32_t notes_unconfigured;
  uint32_t bad_payload;
  uint32_t tx_frames;
  uint32_t max_step_us;
};

class Engine {
 public:
  Engine();
  void begin(WriteFn write, MicrosFn micros, InfoFn info, void* ctx, const char* fw_name, const char* fw_version);
  void set_passthrough(FrameCallback cb, void* cb_ctx);

  // Bytes that arrived from the host.
  void feed(const uint8_t* data, size_t n);
  // Call regularly while no bytes arrive: gives up on a partial frame after STALE_US of silence.
  void idle();
  // Send a frame to the host (also used by the sketch for STATUS, MIC_AUDIO, ...).
  void send(uint8_t type, uint8_t seq, const uint8_t* payload, uint16_t len);
  // Send a LOG text frame (truncated to MAX_TEXT).
  void log(const char* text);

  const EngineStats& stats() const { return st_; }
  const ParserStats& parser_stats() const { return parser_.stats(); }
  const Harmonizer* plugin() const { return h_; }

  static const uint32_t STALE_US = 100000;  // 100 ms

 private:
  static void on_frame_static(const Frame& f, void* self);
  void on_frame(const Frame& f);
  void on_hello(const Frame& f);
  void on_config(const Frame& f);
  void on_note(const Frame& f);

  Parser parser_;
  WriteFn write_;
  MicrosFn micros_;
  InfoFn info_;
  void* ctx_;
  FrameCallback pass_;
  void* pass_ctx_;
  const char* fw_name_;
  const char* fw_version_;
  Harmonizer* h_;
  Meta meta_;
  uint8_t pos_[MAX_POS];
  uint32_t last_rx_us_;
  uint8_t seq_log_;
  EngineStats st_;
  uint8_t tx_[HEADER_LEN + 480 + 1];  // the largest frame a device sends (MIC_AUDIO)
};

}  // namespace duet
