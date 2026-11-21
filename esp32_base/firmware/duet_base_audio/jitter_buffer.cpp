#include "jitter_buffer.h"

namespace duet {

void JitterBuffer::reset() {
  head_.store(0);
  tail_.store(0);
  underruns.store(0);
  overflows.store(0);
  trims.store(0);
  frames_played.store(0);
  state_ = PRIMING;
  last_sample_ = 0;
  fade_in_ = false;
  high_count_ = 0;
}

bool JitterBuffer::push(const int16_t* frame) {
  uint32_t h = head_.load(std::memory_order_relaxed);
  uint32_t t = tail_.load(std::memory_order_acquire);
  if (h - t >= SPK_RING_FRAMES) {
    overflows.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  memcpy(ring_[h % SPK_RING_FRAMES], frame, FRAME_SAMPLES * sizeof(int16_t));
  head_.store(h + 1, std::memory_order_release);
  return true;
}

bool JitterBuffer::pop(int16_t* out) {
  uint32_t t = tail_.load(std::memory_order_relaxed);
  uint32_t h = head_.load(std::memory_order_acquire);
  uint32_t lvl = h - t;

  if (state_ == PRIMING) {
    if (lvl >= SPK_PRIME_FRAMES) {
      state_ = PLAYING;
      fade_in_ = true;
      high_count_ = 0;
    }
  } else if (lvl == 0) {
    underruns.fetch_add(1, std::memory_order_relaxed);
    state_ = PRIMING;
  }

  if (state_ != PLAYING) {
    // Silence, with a short ramp from the last emitted sample to 0 (anti-click).
    for (int i = 0; i < FRAME_SAMPLES; ++i) {
      out[i] = (i < SPK_FADE_SAMPLES) ? (int16_t)((int32_t)last_sample_ * (SPK_FADE_SAMPLES - 1 - i) / SPK_FADE_SAMPLES) : 0;
    }
    last_sample_ = 0;
    return false;
  }

  // Latency trim: level persistently above target -> skip one frame.
  if (lvl > SPK_TRIM_ABOVE) {
    if (++high_count_ >= SPK_TRIM_AFTER) {
      high_count_ = 0;
      ++t;
      --lvl;
      trims.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    high_count_ = 0;
  }

  memcpy(out, ring_[t % SPK_RING_FRAMES], FRAME_SAMPLES * sizeof(int16_t));
  tail_.store(t + 1, std::memory_order_release);
  if (fade_in_) {
    for (int i = 0; i < SPK_FADE_SAMPLES; ++i) out[i] = (int16_t)((int32_t)out[i] * (i + 1) / (SPK_FADE_SAMPLES + 1));
    fade_in_ = false;
  }
  last_sample_ = out[FRAME_SAMPLES - 1];
  frames_played.fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace duet
