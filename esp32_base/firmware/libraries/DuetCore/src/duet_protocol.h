// Duet USB framed protocol, version 2. Portable C++: no Arduino/ESP dependencies, so the same file runs on
// the board, in the Mac simulator (host_tests/board_sim.cpp) and in the host unit tests.
//
// Frame: A5 5A | type u8 | seq u8 | len u16 LE | payload[len] | sum u8
// sum = (type + seq + len_lo + len_hi + sum(payload)) mod 256
//
// Version 1 is the table in duet_v0/docs/PLAN.md (audio, servo, LED, ping, status). Version 2 keeps every
// version-1 type code and length, and adds the note path (HELLO/INFO, CONFIG/CONFIG_ACK, NOTE_IN/NOTE_OUT),
// text logs and AUDIO_CTRL. Types with a variable length are new in version 2.
// Added within version 2 (2026-11-21): AUDIO_QUERY / AUDIO_STATS, the audio counters. A firmware without audio
// (sketch 1) parses AUDIO_QUERY and ignores it, so the host sees no reply.
// Added within version 2 (2026-11-22): the live chain of sketch duet_live (LIVE_CTRL, SYNTH_NOTE, LIVE_NOTE,
// LIVE_STATE, MIC_HOP). Purely additive; a firmware without the live chain parses them and ignores them.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace duet {

enum : uint8_t {
  SYNC0 = 0xA5,
  SYNC1 = 0x5A,
  // ---- version 1 (unchanged)
  T_MIC_AUDIO = 0x01,   // dev->host 240 x int16
  T_SPK_AUDIO = 0x02,   // host->dev 240 x int16
  T_SERVO = 0x10,       // host->dev angle u8, speed u8
  T_LED = 0x11,         // host->dev state u8 (0 off, 1 on, 2 blink)
  T_PING = 0x12,        // host->dev u32 host us
  T_STATUS = 0x20,      // dev->host spk_frames u8, mic_overruns u16, spk_underruns u16
  T_DIAG = 0x21,        // dev->host 8 x u16 (optional)
  T_PONG = 0x22,        // dev->host u32 echoed, u32 device micros
  // ---- version 2 (new)
  T_HELLO = 0x13,       // host->dev empty. The device answers INFO.
  T_CONFIG = 0x14,      // host->dev plugin u8, tonic u8, M u8, flags u8, K u16, n_pos u16, pos[n_pos] u8
  T_NOTE_IN = 0x15,     // host->dev i u16, midi u8, flags u8 (bit0: prev_dur valid), onset f32, prev_dur f32
  T_AUDIO_CTRL = 0x16,  // host->dev flags u8 (bit0: send MIC_AUDIO, bit1: play SPK_AUDIO). Audio sketch only.
  T_AUDIO_QUERY = 0x17, // host->dev empty. The audio sketch (and the simulator) answer AUDIO_STATS, same seq.
  T_INFO = 0x23,        // dev->host ASCII "key=value;key=value", 1..240 bytes
  T_NOTE_OUT = 0x24,    // dev->host i u16, midi i8 (-1 = silent), status u8, compute_us u32
  T_LOG = 0x25,         // dev->host ASCII text, 1..240 bytes
  T_CONFIG_ACK = 0x26,  // dev->host status u8 (0 = ok), plugin u8, n_pos u16 (accepted)
  T_AUDIO_STATS = 0x27, // dev->host AUDIO_STATS_LEN bytes, layout below. Answers AUDIO_QUERY (same seq).
  // ---- version 2, live chain (2026-11-22, sketch duet_live; layouts below)
  T_LIVE_CTRL = 0x18,   // host->dev flags u8, volume u8, bpm u8, reserved u8. Answered by LIVE_STATE (same seq, LS_REPLY)
  T_SYNTH_NOTE = 0x19,  // host->dev midi u8, vel u8 (0 = note off). No reply.
  T_LIVE_NOTE = 0x28,   // dev->host LIVE_NOTE_LEN bytes: one heard / sung note event
  T_LIVE_STATE = 0x29,  // dev->host LIVE_STATE_LEN bytes: state of the live chain
  T_MIC_HOP = 0x2A,     // dev->host 256 x int16: one hop of what the pitch tracker got (LC_MONITOR only)
};

static const uint8_t PROTO_VERSION = 2;
static const size_t HEADER_LEN = 6;       // sync(2) type seq len(2)
static const size_t MAX_TEXT = 240;       // INFO and LOG payload cap
static const size_t CONFIG_HEADER_LEN = 8;
static const size_t MAX_POS = 1000;       // beats of bar-position table a CONFIG may carry
static const size_t MAX_PAYLOAD = CONFIG_HEADER_LEN + MAX_POS;  // 1008, the largest legal payload
static const size_t MAX_FRAME = HEADER_LEN + MAX_PAYLOAD + 1;

static const uint16_t NOTE_IN_LEN = 12;
static const uint16_t NOTE_OUT_LEN = 8;
static const uint16_t CONFIG_ACK_LEN = 4;
static const uint16_t AUDIO_STATS_LEN = 48;

// AUDIO_STATS payload (little-endian). Counters are u32 since boot and wrap.
//   0  mode u8          AUDIO_MODE_*: 1 = real I2S hardware, 2 = stand-in (no audio hardware driven)
//   1  flags u8         AS_MIC_ON | AS_SPK_ON (the AUDIO_CTRL gates) | AS_RUNNING (audio started) |
//                       AS_SPK_SUMS (spk_silent, spk_crc, spk_peak are kept; the stand-in speaker keeps them)
//   2  spk_level u8     frames waiting in the speaker jitter buffer
//   3  spk_state u8     jitter buffer: 0 = priming, 1 = playing
//   4  mic_sent u32     MIC_AUDIO frames written to the host
//   8  mic_dropped u32  mic frames lost on the device (queue full, USB buffer full, read errors, DMA overflows)
//  12  spk_rx u32       SPK_AUDIO frames put into the jitter buffer
//  16  spk_played u32   audio frames taken out of the jitter buffer and played
//  20  spk_silent u32   silent frames played (jitter buffer priming or empty)
//  24  spk_underruns u32  jitter buffer underruns
//  28  spk_dropped u32  SPK_AUDIO frames not played: ring full, latency trim, or arrived with the gate off
//  32  spk_seq_gaps u32 SPK_AUDIO seq discontinuities
//  36  spk_crc u32      CRC-32 (zlib) of the int16 LE bytes of every sample of every played audio frame
//  40  spk_peak u16     max |sample| of the played audio frames
//  42  reserved u16     0
//  44  dev_us u32       device clock when the reply was made
enum : uint8_t { AUDIO_MODE_NONE = 0, AUDIO_MODE_I2S = 1, AUDIO_MODE_STANDIN = 2 };
enum : uint8_t { AS_MIC_ON = 0x01, AS_SPK_ON = 0x02, AS_RUNNING = 0x04, AS_SPK_SUMS = 0x08 };

struct AudioStats {
  uint8_t mode, flags, spk_level, spk_state;
  uint32_t mic_sent, mic_dropped, spk_rx, spk_played, spk_silent, spk_underruns, spk_dropped, spk_seq_gaps;
  uint32_t spk_crc;
  uint16_t spk_peak;
  uint32_t dev_us;
};
void pack_audio_stats(const AudioStats& s, uint8_t* out);  // out has AUDIO_STATS_LEN bytes
bool parse_audio_stats(const uint8_t* p, uint16_t len, AudioStats* s);

// ------------------------------------------------------------------ live chain (sketch duet_live)
// LIVE_CTRL  host->dev, 4 bytes: flags u8 (LC_*), volume u8 (0..100 % of the synth's 0.35 cap; > 100 = 100),
//            bpm u8 (only to turn seconds into beats for the plug-in; 0 = 120), reserved u8 (0).
//            The device answers LIVE_STATE with the same seq and LS_REPLY set.
// SYNTH_NOTE host->dev, 2 bytes: midi u8, vel u8 (0 = off). Ignored unless SING is on. Reported as a SUNG event
//            with LN_FROM_HOST.
// LIVE_NOTE  dev->host, 8 bytes: kind u8 (LN_*), midi u8, vel u8, flags u8 (LN_FAKE | LN_FROM_HOST),
//            t_ms u32 (hops since LISTEN went on x 16; 0 when LISTEN is off). seq: own counter, 0 at each
//            rising edge of LISTEN.
// LIVE_STATE dev->host, 24 bytes:
//    0 flags u8 (LS_*)   1 volume u8   2 heard i8 (-1 none)   3 sung i8 (-1 none)   4 freq f32 (Hz, 0 unvoiced)
//    8 conf u8 (0..100)  9 level i8 (dB, -100..0)  10 cents i8 (-50..50)  11 reserved u8
//   12 hops u32  16 heard_n u16  18 sung_n u16  20 spk_blocks u32
//   Sent every LIVE_STATE_EVERY hops while LISTEN is on (own seq, LS_REPLY clear) and as the LIVE_CTRL answer.
// MIC_HOP    dev->host, 512 bytes: 256 int16 LE, the hop as the tracker got it before its DC filter.
//            seq +1 per hop, 0 at each rising edge of LISTEN. LC_MONITOR only.
static const uint16_t LIVE_CTRL_LEN = 4;
static const uint16_t SYNTH_NOTE_LEN = 2;
static const uint16_t LIVE_NOTE_LEN = 8;
static const uint16_t LIVE_STATE_LEN = 24;
static const uint16_t MIC_HOP_SAMPLES = 256;
static const uint16_t MIC_HOP_LEN = 2 * MIC_HOP_SAMPLES;  // 512
static const uint32_t LIVE_STATE_EVERY = 6;                // hops (96 ms)
static const uint32_t LIVE_HOP_MS = 16;                    // 256 samples at 16 kHz
// The largest payload a device sends (MIC_HOP). Engine's transmit buffer is sized for it.
static const uint16_t MAX_DEVICE_PAYLOAD = MIC_HOP_LEN;

enum : uint8_t { LC_LISTEN = 0x01, LC_SING = 0x02, LC_FAKE = 0x04, LC_MONITOR = 0x08, LC_KNOWN = 0x0F };
enum : uint8_t { LN_HEARD_ON = 1, LN_HEARD_OFF = 2, LN_SUNG_ON = 3, LN_SUNG_OFF = 4 };
enum : uint8_t { LN_FAKE = 0x01, LN_FROM_HOST = 0x02 };
enum : uint8_t {
  LS_LISTEN = 0x01, LS_SING = 0x02, LS_FAKE = 0x04, LS_MONITOR = 0x08,
  LS_MIC_UP = 0x10, LS_SPK_UP = 0x20, LS_ERROR = 0x40, LS_REPLY = 0x80,
};

struct LiveCtrl {
  uint8_t flags, volume, bpm;
};
struct LiveNote {
  uint8_t kind, midi, vel, flags;
  uint32_t t_ms;
};
struct LiveState {
  uint8_t flags, volume;
  int8_t heard, sung;
  float freq;
  uint8_t conf;
  int8_t level, cents;
  uint32_t hops;
  uint16_t heard_n, sung_n;
  uint32_t spk_blocks;
};
void pack_live_ctrl(const LiveCtrl& c, uint8_t* out);  // LIVE_CTRL_LEN bytes
bool parse_live_ctrl(const uint8_t* p, uint16_t len, LiveCtrl* c);
void pack_live_note(const LiveNote& n, uint8_t* out);  // LIVE_NOTE_LEN bytes
bool parse_live_note(const uint8_t* p, uint16_t len, LiveNote* n);
void pack_live_state(const LiveState& s, uint8_t* out);  // LIVE_STATE_LEN bytes
bool parse_live_state(const uint8_t* p, uint16_t len, LiveState* s);

// Allowed payload length range of a type. Returns false if the type is unknown.
bool len_range(uint8_t type, uint16_t* min_len, uint16_t* max_len);

uint8_t checksum(uint8_t type, uint8_t seq, uint16_t len, const uint8_t* payload);

// Encode one frame into out (capacity >= HEADER_LEN + len + 1). Returns bytes written.
size_t encode(uint8_t type, uint8_t seq, const uint8_t* payload, uint16_t len, uint8_t* out);

struct Frame {
  uint8_t type;
  uint8_t seq;
  uint16_t len;
  const uint8_t* payload;  // valid only during the callback
};

struct ParserStats {
  uint32_t frames_ok;
  uint32_t bad_checksum;
  uint32_t bad_header;     // sync found but unknown type / length outside the type's range
  uint32_t bytes_skipped;  // bytes discarded while hunting for sync
  uint32_t stale_drops;    // partial frames given up after a quiet link (drop_stale)
};

typedef void (*FrameCallback)(const Frame& f, void* ctx);

// Streaming parser with resynchronisation. On any error (bad header, bad checksum) it discards exactly ONE
// byte (the first sync byte) and rescans, so a valid frame that starts inside the bytes of a broken one is
// still found. Unknown types are rejected.
class Parser {
 public:
  Parser();
  void reset();
  // Feed n bytes; cb is called for each complete valid frame.
  void feed(const uint8_t* data, size_t n, FrameCallback cb, void* ctx);
  // Bytes waiting for the rest of a frame.
  size_t pending() const { return n_; }
  // A corrupted header of a variable-length type can announce more bytes than will ever come. When the link
  // has been quiet, call this: it gives up on the first pending byte and rescans what is left.
  void drop_stale(FrameCallback cb, void* ctx);
  const ParserStats& stats() const { return st_; }

 private:
  void process(FrameCallback cb, void* ctx);
  uint8_t buf_[2 * MAX_FRAME];
  size_t n_;
  ParserStats st_;
};

// Little-endian helpers.
static inline void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void put_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t get_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

}  // namespace duet
