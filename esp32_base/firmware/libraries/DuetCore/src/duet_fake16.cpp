#include "duet_fake16.h"

#include "duet_standin.h"  // standin_psin: the same parabolic sine (a function, no 24 kHz table)

namespace duet {

const Fake16Note FAKE16_MELODY[FAKE16_NOTES] = {
    {60, 70229578u, FAKE16_SHORT}, {60, 70229578u, FAKE16_SHORT}, {67, 105225474u, FAKE16_SHORT},
    {67, 105225474u, FAKE16_SHORT}, {69, 118111601u, FAKE16_SHORT}, {69, 118111601u, FAKE16_SHORT},
    {67, 105225474u, FAKE16_LONG},  {65, 93745240u, FAKE16_SHORT},  {65, 93745240u, FAKE16_SHORT},
    {64, 88483724u, FAKE16_SHORT},  {64, 88483724u, FAKE16_SHORT},  {62, 78830036u, FAKE16_SHORT},
    {62, 78830036u, FAKE16_SHORT},  {60, 70229578u, FAKE16_LONG},
};

int16_t fake16_sample(uint32_t n, uint32_t gap) {
  uint32_t pos = n % FAKE16_LOOP;
  size_t k = 0;
  while (k + 1 < FAKE16_NOTES && pos >= FAKE16_MELODY[k].slot) {
    pos -= FAKE16_MELODY[k].slot;
    ++k;
  }
  const Fake16Note& note = FAKE16_MELODY[k];
  const uint32_t t = pos;  // samples since this note started
  const uint32_t sound = note.slot - gap;
  if (t >= sound) return 0;
  int64_t env;  // Q15, 0..32768
  if (t < FAKE16_ATTACK) {
    env = ((int64_t)t * 32768) / FAKE16_ATTACK;
  } else if (t >= sound - FAKE16_RELEASE) {
    env = ((int64_t)(sound - t) * 32768) / FAKE16_RELEASE;
  } else {
    env = 32768;
  }
  const uint32_t ph = t * note.inc;  // wraps mod 2^32 (unsigned)
  const int32_t mix = STANDIN_W1 * standin_psin(ph) + STANDIN_W2 * standin_psin(ph * 2u) +
                      STANDIN_W3 * standin_psin(ph * 3u);
  const int64_t mag = mix < 0 ? -(int64_t)mix : (int64_t)mix;
  const int64_t v = (mag * env * STANDIN_GAIN_MUL) >> STANDIN_GAIN_SHIFT;
  return (int16_t)(mix < 0 ? -v : v);
}

void fake16_fill(uint32_t n0, int16_t* out, size_t count, uint32_t gap) {
  for (size_t i = 0; i < count; ++i) out[i] = fake16_sample(n0 + (uint32_t)i, gap);
}

}  // namespace duet
