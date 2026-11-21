#include "duet_standin.h"

namespace duet {

const StandinNote STANDIN_MELODY[STANDIN_NOTES] = {
    {60, 46819719u, 12000}, {60, 46819719u, 12000}, {67, 70150316u, 12000}, {67, 70150316u, 12000},
    {69, 78741067u, 12000}, {69, 78741067u, 12000}, {67, 70150316u, 24000}, {65, 62496826u, 12000},
    {65, 62496826u, 12000}, {64, 58989149u, 12000}, {64, 58989149u, 12000}, {62, 52553357u, 12000},
    {62, 52553357u, 12000}, {60, 46819719u, 24000},
};

int32_t standin_psin(uint32_t phase) {
  const uint32_t p = phase >> 16;
  const int32_t x = (int32_t)(p & 0x7FFF);
  const int32_t y = (x * (32768 - x)) >> 14;  // x * (32768 - x) <= 2^28, and >= 0
  return (p & 0x8000) ? -y : y;
}

int16_t standin_mic_sample(uint32_t n) {
  uint32_t pos = n % STANDIN_LOOP;
  size_t k = 0;
  while (k + 1 < STANDIN_NOTES && pos >= STANDIN_MELODY[k].slot) {
    pos -= STANDIN_MELODY[k].slot;
    ++k;
  }
  const StandinNote& note = STANDIN_MELODY[k];
  const uint32_t t = pos;  // samples since this note started
  const uint32_t sound = note.slot - STANDIN_GAP;
  if (t >= sound) return 0;
  int64_t env;  // Q15, 0..32768
  if (t < STANDIN_ATTACK) {
    env = ((int64_t)t * 32768) / STANDIN_ATTACK;
  } else if (t >= sound - STANDIN_RELEASE) {
    env = ((int64_t)(sound - t) * 32768) / STANDIN_RELEASE;
  } else {
    env = 32768;
  }
  const uint32_t ph = t * note.inc;  // wraps mod 2^32 (unsigned)
  const int32_t mix = STANDIN_W1 * standin_psin(ph) + STANDIN_W2 * standin_psin(ph * 2u) +
                      STANDIN_W3 * standin_psin(ph * 3u);
  const int64_t mag = mix < 0 ? -(int64_t)mix : (int64_t)mix;
  const int64_t v = (mag * env * STANDIN_GAIN_MUL) >> STANDIN_GAIN_SHIFT;  // non-negative, < 2^45 before the shift
  return (int16_t)(mix < 0 ? -v : v);
}

void standin_mic_fill(uint32_t n0, int16_t* out, size_t count) {
  for (size_t i = 0; i < count; ++i) out[i] = standin_mic_sample(n0 + (uint32_t)i);
}

uint32_t crc32_samples(uint32_t crc, const int16_t* x, size_t n) {
  uint32_t c = ~crc;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t s = (uint16_t)x[i];
    const uint8_t bytes[2] = {(uint8_t)(s & 0xFF), (uint8_t)(s >> 8)};
    for (uint8_t b : bytes) {
      c ^= b;
      for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
  }
  return ~c;
}

void StandinSpeaker::reset() {
  frames.store(0);
  silent.store(0);
  crc.store(0);
  peak.store(0);
}

void StandinSpeaker::play(const int16_t* frame) {
  uint32_t pk = peak.load(std::memory_order_relaxed);
  for (size_t i = 0; i < STANDIN_FRAME; ++i) {
    const int32_t v = frame[i];
    const uint32_t a = (uint32_t)(v < 0 ? -v : v);
    if (a > pk) pk = a;
  }
  peak.store(pk, std::memory_order_relaxed);
  crc.store(crc32_samples(crc.load(std::memory_order_relaxed), frame, STANDIN_FRAME), std::memory_order_relaxed);
  frames.fetch_add(1, std::memory_order_release);  // last: a reader that sees the new count sees its crc
}

}  // namespace duet
