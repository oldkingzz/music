#include "duet_engine.h"

#include <stdio.h>
#include <string.h>

namespace duet {

Engine::Engine()
    : write_(nullptr), micros_(nullptr), info_(nullptr), ctx_(nullptr), pass_(nullptr), pass_ctx_(nullptr),
      fw_name_("?"), fw_version_("?"), h_(nullptr), last_rx_us_(0), seq_log_(0) {
  memset(&st_, 0, sizeof(st_));
  memset(&meta_, 0, sizeof(meta_));
  memset(pos_, 0, sizeof(pos_));
}

void Engine::begin(WriteFn write, MicrosFn micros, InfoFn info, void* ctx, const char* fw_name,
                   const char* fw_version) {
  write_ = write;
  micros_ = micros;
  info_ = info;
  ctx_ = ctx;
  fw_name_ = fw_name;
  fw_version_ = fw_version;
  last_rx_us_ = micros_ ? micros_() : 0;
}

void Engine::set_passthrough(FrameCallback cb, void* cb_ctx) {
  pass_ = cb;
  pass_ctx_ = cb_ctx;
}

void Engine::feed(const uint8_t* data, size_t n) {
  if (!n) return;
  if (micros_) last_rx_us_ = micros_();
  parser_.feed(data, n, on_frame_static, this);
}

void Engine::idle() {
  if (!micros_ || !parser_.pending()) return;
  const uint32_t now = micros_();
  if ((uint32_t)(now - last_rx_us_) < STALE_US) return;
  last_rx_us_ = now;  // one byte per STALE_US of silence
  parser_.drop_stale(on_frame_static, this);
}

void Engine::send(uint8_t type, uint8_t seq, const uint8_t* payload, uint16_t len) {
  if (!write_ || (size_t)HEADER_LEN + len + 1 > sizeof(tx_)) return;
  const size_t n = encode(type, seq, payload, len, tx_);
  ++st_.tx_frames;
  write_(tx_, n, ctx_);
}

void Engine::log(const char* text) {
  size_t n = strlen(text);
  if (!n) return;
  if (n > MAX_TEXT) n = MAX_TEXT;
  send(T_LOG, seq_log_++, (const uint8_t*)text, (uint16_t)n);
}

void Engine::on_frame_static(const Frame& f, void* self) { ((Engine*)self)->on_frame(f); }

void Engine::on_frame(const Frame& f) {
  switch (f.type) {
    case T_PING: {
      ++st_.pings;
      uint8_t p[8];
      memcpy(p, f.payload, 4);
      put_u32(p + 4, micros_ ? micros_() : 0);
      send(T_PONG, f.seq, p, 8);
      break;
    }
    case T_HELLO: on_hello(f); break;
    case T_CONFIG: on_config(f); break;
    case T_NOTE_IN: on_note(f); break;
    default:
      if (pass_) pass_(f, pass_ctx_);
      break;
  }
}

void Engine::on_hello(const Frame& f) {
  ++st_.hellos;
  char text[MAX_TEXT + 1];
  int n = snprintf(text, sizeof(text), "fw=%s;ver=%s;proto=%u;plugins=", fw_name_, fw_version_,
                   (unsigned)PROTO_VERSION);
  for (uint8_t id = 0; id < H_COUNT && n < (int)MAX_TEXT; ++id) {
    const Harmonizer* h = harmonizer(id);
    n += snprintf(text + n, sizeof(text) - n, "%s%s", id ? "," : "", h ? h->name() : "?");
  }
  if (n > (int)MAX_TEXT) n = MAX_TEXT;
  if (info_ && n + 1 < (int)MAX_TEXT) {
    text[n++] = ';';
    const size_t cap = MAX_TEXT - (size_t)n;  // snprintf-style: at most cap - 1 characters plus a NUL
    size_t k = info_(text + n, cap, ctx_);
    if (k >= cap) k = cap - 1;
    n += (int)k;
  }
  send(T_INFO, f.seq, (const uint8_t*)text, (uint16_t)n);
}

void Engine::on_config(const Frame& f) {
  ++st_.configs;
  ConfigMsg c;
  uint8_t ack[CONFIG_ACK_LEN] = {ACK_OK, 0, 0, 0};
  if (!parse_config(f.payload, f.len, &c)) {
    ++st_.bad_payload;
    ack[0] = ACK_BAD_CONFIG;
    ack[1] = f.len ? f.payload[0] : 0;
    h_ = nullptr;
    send(T_CONFIG_ACK, f.seq, ack, CONFIG_ACK_LEN);
    return;
  }
  ack[1] = c.plugin;
  Harmonizer* h = harmonizer(c.plugin);
  if (!h) {
    ack[0] = ACK_UNKNOWN_PLUGIN;
    h_ = nullptr;
    send(T_CONFIG_ACK, f.seq, ack, CONFIG_ACK_LEN);
    return;
  }
  memcpy(pos_, c.pos, c.n_pos);  // the frame buffer is reused; the plug-in keeps a pointer to our copy
  meta_.tonic = c.tonic;
  meta_.M = c.M;
  meta_.K = c.K;
  meta_.pos = pos_;
  meta_.n_pos = c.n_pos;
  h_ = h;
  h_->start(meta_);
  put_u16(ack + 2, c.n_pos);
  send(T_CONFIG_ACK, f.seq, ack, CONFIG_ACK_LEN);
}

static_assert(MAX_DEVICE_PAYLOAD >= 480 && MAX_DEVICE_PAYLOAD >= MIC_HOP_LEN && MAX_DEVICE_PAYLOAD >= MAX_TEXT &&
                  MAX_DEVICE_PAYLOAD >= LIVE_STATE_LEN,
              "Engine::send must fit every frame a device sends");

int Engine::plugin_step(int i, float onset, int midi, float prev_dur) {
  if (!h_) return -1;
  int m = h_->step(i, onset, midi, prev_dur);
  if (m < 0 || m > 127) m = -1;
  return m;
}

void Engine::on_note(const Frame& f) {
  NoteIn in;
  if (!parse_note_in(f.payload, f.len, &in)) {
    ++st_.bad_payload;
    return;
  }
  NoteOut out;
  out.i = in.i;
  out.midi = -1;
  out.status = NOTE_OK;
  out.compute_us = 0;
  if (!h_) {
    ++st_.notes_unconfigured;
    out.status = NOTE_NOT_CONFIGURED;
  } else {
    ++st_.notes;
    const uint32_t t0 = micros_ ? micros_() : 0;
    int m = h_->step(in.i, in.onset, in.midi, in.has_prev ? in.prev_dur : -1.0f);
    const uint32_t dt = micros_ ? (uint32_t)(micros_() - t0) : 0;
    if (m < 0 || m > 127) m = -1;
    out.midi = (int8_t)m;
    out.compute_us = dt;
    if (dt > st_.max_step_us) st_.max_step_us = dt;
  }
  uint8_t p[NOTE_OUT_LEN];
  pack_note_out(out, p);
  send(T_NOTE_OUT, f.seq, p, NOTE_OUT_LEN);
}

}  // namespace duet
