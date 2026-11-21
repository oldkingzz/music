// Speaker jitter buffer: single-producer (USB RX task, core 1) / single-consumer (speaker
// task, core 0) lock-free ring of 10 ms frames, plus the playback policy:
//   * PRIMING until SPK_PRIME_FRAMES are queued, then PLAYING.
//   * PLAYING and ring empty -> underrun: count it once, output silence (with a 1 ms fade
//     from the last sample to avoid a click) and go back to PRIMING.
//   * push into a full ring -> the incoming frame is dropped (overflow counter). Dropping
//     the newest keeps the ring strictly SPSC (only the consumer moves the tail).
//   * if the level stays above SPK_TRIM_ABOVE for SPK_TRIM_AFTER consecutive pops, the
//     consumer drops one frame (bounded latency if the host clock runs fast).
// Portable C++11 (std::atomic) so it is unit-tested on the host.
#pragma once
#include <stdint.h>
#include <string.h>

#include <atomic>

#include "config.h"

namespace duet {

class JitterBuffer {
 public:
  enum State : uint8_t { PRIMING = 0, PLAYING = 1 };

  JitterBuffer() { reset(); }
  void reset();

  // Producer side. Returns false (and counts an overflow) if the ring is full.
  bool push(const int16_t* frame240);

  // Consumer side: always fills out[240] (audio or silence). Returns true if real audio.
  bool pop(int16_t* out240);

  uint8_t level() const {
    uint32_t h = head_.load(std::memory_order_acquire), t = tail_.load(std::memory_order_acquire);
    return (uint8_t)(h - t);
  }
  State state() const { return state_; }

  // Counters (wrap-around, read from any task; written by exactly one side each).
  std::atomic<uint32_t> underruns;  // consumer
  std::atomic<uint32_t> overflows;  // producer
  std::atomic<uint32_t> trims;      // consumer
  std::atomic<uint32_t> frames_played;

 private:
  int16_t ring_[SPK_RING_FRAMES][FRAME_SAMPLES];
  std::atomic<uint32_t> head_;  // written by producer only
  std::atomic<uint32_t> tail_;  // written by consumer only
  State state_;                 // consumer only
  int16_t last_sample_;         // consumer only
  bool fade_in_;                // consumer only
  uint32_t high_count_;         // consumer only
};

}  // namespace duet
