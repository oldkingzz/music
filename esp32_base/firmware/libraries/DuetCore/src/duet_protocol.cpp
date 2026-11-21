#include "duet_protocol.h"

#include <string.h>

namespace duet {

bool len_range(uint8_t type, uint16_t* min_len, uint16_t* max_len) {
  uint16_t lo, hi;
  switch (type) {
    case T_MIC_AUDIO: lo = hi = 480; break;
    case T_SPK_AUDIO: lo = hi = 480; break;
    case T_SERVO: lo = hi = 2; break;
    case T_LED: lo = hi = 1; break;
    case T_PING: lo = hi = 4; break;
    case T_STATUS: lo = hi = 5; break;
    case T_DIAG: lo = hi = 16; break;
    case T_PONG: lo = hi = 8; break;
    case T_HELLO: lo = hi = 0; break;
    case T_CONFIG: lo = CONFIG_HEADER_LEN; hi = (uint16_t)(CONFIG_HEADER_LEN + MAX_POS); break;
    case T_NOTE_IN: lo = hi = NOTE_IN_LEN; break;
    case T_AUDIO_CTRL: lo = hi = 1; break;
    case T_INFO: lo = 1; hi = MAX_TEXT; break;
    case T_NOTE_OUT: lo = hi = NOTE_OUT_LEN; break;
    case T_LOG: lo = 1; hi = MAX_TEXT; break;
    case T_CONFIG_ACK: lo = hi = CONFIG_ACK_LEN; break;
    case T_AUDIO_QUERY: lo = hi = 0; break;
    case T_AUDIO_STATS: lo = hi = AUDIO_STATS_LEN; break;
    default: return false;
  }
  if (min_len) *min_len = lo;
  if (max_len) *max_len = hi;
  return true;
}

uint8_t checksum(uint8_t type, uint8_t seq, uint16_t len, const uint8_t* payload) {
  uint32_t s = (uint32_t)type + seq + (len & 0xFF) + (len >> 8);
  for (uint16_t i = 0; i < len; ++i) s += payload[i];
  return (uint8_t)s;
}

void pack_audio_stats(const AudioStats& s, uint8_t* out) {
  out[0] = s.mode;
  out[1] = s.flags;
  out[2] = s.spk_level;
  out[3] = s.spk_state;
  put_u32(out + 4, s.mic_sent);
  put_u32(out + 8, s.mic_dropped);
  put_u32(out + 12, s.spk_rx);
  put_u32(out + 16, s.spk_played);
  put_u32(out + 20, s.spk_silent);
  put_u32(out + 24, s.spk_underruns);
  put_u32(out + 28, s.spk_dropped);
  put_u32(out + 32, s.spk_seq_gaps);
  put_u32(out + 36, s.spk_crc);
  put_u16(out + 40, s.spk_peak);
  put_u16(out + 42, 0);
  put_u32(out + 44, s.dev_us);
}

bool parse_audio_stats(const uint8_t* p, uint16_t len, AudioStats* s) {
  if (len != AUDIO_STATS_LEN || !s) return false;
  s->mode = p[0];
  s->flags = p[1];
  s->spk_level = p[2];
  s->spk_state = p[3];
  s->mic_sent = get_u32(p + 4);
  s->mic_dropped = get_u32(p + 8);
  s->spk_rx = get_u32(p + 12);
  s->spk_played = get_u32(p + 16);
  s->spk_silent = get_u32(p + 20);
  s->spk_underruns = get_u32(p + 24);
  s->spk_dropped = get_u32(p + 28);
  s->spk_seq_gaps = get_u32(p + 32);
  s->spk_crc = get_u32(p + 36);
  s->spk_peak = get_u16(p + 40);
  s->dev_us = get_u32(p + 44);
  return true;
}

size_t encode(uint8_t type, uint8_t seq, const uint8_t* payload, uint16_t len, uint8_t* out) {
  out[0] = SYNC0;
  out[1] = SYNC1;
  out[2] = type;
  out[3] = seq;
  put_u16(out + 4, len);
  if (len) memcpy(out + HEADER_LEN, payload, len);
  out[HEADER_LEN + len] = checksum(type, seq, len, payload);
  return HEADER_LEN + len + 1;
}

Parser::Parser() : n_(0) { memset(&st_, 0, sizeof(st_)); }

void Parser::reset() { n_ = 0; }

void Parser::feed(const uint8_t* data, size_t n, FrameCallback cb, void* ctx) {
  while (n) {
    size_t room = sizeof(buf_) - n_;
    size_t k = n < room ? n : room;
    memcpy(buf_ + n_, data, k);
    n_ += k;
    data += k;
    n -= k;
    process(cb, ctx);
  }
}

void Parser::drop_stale(FrameCallback cb, void* ctx) {
  if (!n_) return;
  memmove(buf_, buf_ + 1, n_ - 1);
  --n_;
  ++st_.bytes_skipped;
  ++st_.stale_drops;
  process(cb, ctx);
}

// Consume as many complete frames as possible from buf_[0..n_).
void Parser::process(FrameCallback cb, void* ctx) {
  size_t pos = 0;
  for (;;) {
    // 1) hunt for sync
    while (pos < n_ && !(buf_[pos] == SYNC0 && (pos + 1 >= n_ || buf_[pos + 1] == SYNC1))) {
      ++pos;
      ++st_.bytes_skipped;
    }
    if (pos + HEADER_LEN > n_) break;  // need more bytes (possibly a lone trailing A5)
    const uint8_t type = buf_[pos + 2];
    const uint16_t len = get_u16(buf_ + pos + 4);
    uint16_t lo = 0, hi = 0;
    const bool header_ok = len_range(type, &lo, &hi) && len >= lo && len <= hi;
    if (!header_ok) {
      ++st_.bad_header;
      ++pos;  // drop one byte and rescan
      ++st_.bytes_skipped;
      continue;
    }
    const size_t flen = HEADER_LEN + len + 1;
    if (pos + flen > n_) break;  // wait for the rest
    const uint8_t* p = buf_ + pos;
    if (checksum(type, p[3], len, p + HEADER_LEN) != p[HEADER_LEN + len]) {
      ++st_.bad_checksum;
      ++pos;
      ++st_.bytes_skipped;
      continue;
    }
    ++st_.frames_ok;
    Frame f;
    f.type = type;
    f.seq = p[3];
    f.len = len;
    f.payload = p + HEADER_LEN;
    if (cb) cb(f, ctx);
    pos += flen;
  }
  if (pos) {
    memmove(buf_, buf_ + pos, n_ - pos);
    n_ -= pos;
  }
}

}  // namespace duet
