// 24 kHz -> 48 kHz half-band interpolator (polyphase). Portable, host-tested.
// Cost: 16 multiplies + 32 adds per input sample (symmetric pre-add) -> ~0.4 M MAC/s.
#pragma once
#include <stdint.h>

#include "halfband_coefs.h"

namespace duet {

class Upsampler2x {
 public:
  Upsampler2x() { reset(); }
  void reset();
  // in[n] mono int16 at 24 kHz -> out[2n] mono int16 at 48 kHz.
  void process(const int16_t* in, int n, int16_t* out);
  // Same, but writes each output sample to both slots of an interleaved stereo buffer
  // (out has 4n int16), which is what the MAX98357A I2S stream needs.
  void process_stereo(const int16_t* in, int n, int16_t* out_lr);

 private:
  inline int16_t sat(float v) {
    if (v > 32767.f) return 32767;
    if (v < -32768.f) return -32768;
    return (int16_t)(v >= 0 ? v + 0.5f : v - 0.5f);
  }
  // Circular history of the last HB_EVEN_LEN inputs, stored twice so the FIR can read a
  // contiguous window without wrap checks.
  float hist_[2 * HB_EVEN_LEN];
  int idx_;
};

}  // namespace duet
