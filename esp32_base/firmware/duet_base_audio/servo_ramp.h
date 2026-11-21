// Speed-limited servo set-point generator (portable, host-tested).
// SERVO payload: angle u8 (0..180 deg on the wire, clamped to SERVO_MIN_DEG..SERVO_MAX_DEG = rest +/- 20 deg,
// the travel mech/ allows), speed u8 (deg per 10 ms tick, 0 = jump).
// tick() is called every CONTROL_PERIOD_MS = 10 ms; position is kept in centi-degrees.
#pragma once
#include <stdint.h>

#include "config.h"

namespace duet {

class ServoRamp {
 public:
  ServoRamp() : pos_cd_(SERVO_REST_DEG * 100), target_cd_(SERVO_REST_DEG * 100), step_cd_(0), active_(false) {}

  void command(uint8_t angle_deg, uint8_t speed_deg_per_tick) {
    if (angle_deg < SERVO_MIN_DEG) angle_deg = SERVO_MIN_DEG;
    if (angle_deg > SERVO_MAX_DEG) angle_deg = SERVO_MAX_DEG;
    target_cd_ = (int32_t)angle_deg * 100;
    step_cd_ = (int32_t)speed_deg_per_tick * 100;
    if (step_cd_ == 0) pos_cd_ = target_cd_;
    active_ = true;
  }

  // Advance one 10 ms tick. Returns current position (centi-degrees).
  int32_t tick() {
    if (step_cd_ > 0 && pos_cd_ != target_cd_) {
      int32_t d = target_cd_ - pos_cd_;
      if (d > step_cd_) d = step_cd_;
      if (d < -step_cd_) d = -step_cd_;
      pos_cd_ += d;
    }
    return pos_cd_;
  }

  bool active() const { return active_; }        // false until the first SERVO frame
  bool moving() const { return pos_cd_ != target_cd_; }
  int32_t pos_cd() const { return pos_cd_; }

  // 0..180 deg -> 500..2500 us
  static uint32_t cd_to_us(int32_t cd) {
    return SERVO_MIN_US + (uint32_t)(((int64_t)cd * (SERVO_MAX_US - SERVO_MIN_US) + 9000) / 18000);
  }
  // pulse width -> LEDC duty at SERVO_HZ with SERVO_RES_BITS
  static uint32_t us_to_duty(uint32_t us) {
    const uint64_t period_us = 1000000ULL / SERVO_HZ;
    return (uint32_t)(((uint64_t)us * ((1u << SERVO_RES_BITS)) + period_us / 2) / period_us);
  }

 private:
  int32_t pos_cd_, target_cd_, step_cd_;
  bool active_;
};

}  // namespace duet
