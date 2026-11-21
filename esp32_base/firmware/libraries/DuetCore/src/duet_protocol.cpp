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
