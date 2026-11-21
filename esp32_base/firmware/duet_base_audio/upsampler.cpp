#include "upsampler.h"

#include <string.h>

namespace duet {

void Upsampler2x::reset() {
  memset(hist_, 0, sizeof(hist_));
  idx_ = 0;
}

// After inserting x[n] at position idx_ (and idx_+L), the window w[k] = hist_[idx_ + k],
// k = 0..L-1, holds x[n-(L-1)+k]... we store newest at the highest index:
//   w[L-1-j] = x[n-j].
void Upsampler2x::process(const int16_t* in, int n, int16_t* out) {
  const int L = HB_EVEN_LEN;
  for (int i = 0; i < n; ++i) {
    float x = (float)in[i];
    hist_[idx_] = x;
    hist_[idx_ + L] = x;
    idx_ = (idx_ + 1) % L;
    const float* w = &hist_[idx_];  // w[0] = oldest = x[n-(L-1)], w[L-1] = newest = x[n]
    // even phase: y = sum_j g[j] x[n-j] = sum_j g[j] w[L-1-j]; g symmetric -> pre-add
    float acc = 0.f;
    for (int j = 0; j < L / 2; ++j) acc += HB_EVEN[j] * (w[L - 1 - j] + w[j]);
    out[2 * i] = sat(acc);
    // odd phase: pure delay x[n-D] = w[L-1-D]
    out[2 * i + 1] = sat(w[L - 1 - HB_DELAY]);
  }
}

void Upsampler2x::process_stereo(const int16_t* in, int n, int16_t* out_lr) {
  // Reuse process() into the second half of out_lr, then expand in place front-to-back.
  int16_t* mono = out_lr + 2 * n;  // 2n mono samples fit in the upper half (size 4n)
  process(in, n, mono);
  for (int k = 0; k < 2 * n; ++k) {
    int16_t s = mono[k];
    out_lr[2 * k] = s;
    out_lr[2 * k + 1] = s;
  }
}

}  // namespace duet
