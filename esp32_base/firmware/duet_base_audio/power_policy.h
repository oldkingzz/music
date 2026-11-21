// Servo + amplifier rail sequencing (PWR_EN on the PCB). Portable and host-tested.
//
//   * Boot: rail off. The board's 100 k pull-down holds TPS22917 ON low through reset; setup() then
//     drives PWR_EN low explicitly.
//   * Rail on at the first valid host frame. The host can only send frames after the USB device is
//     configured and an application has opened the port, so during enumeration the board draws only
//     the ESP32 and the analog chain (USB 2.0 section 7.2.1.4: come up below one unit load).
//   * Loads ready PWR_RAIL_SETTLE_MS after the rail turned on: only then may AMP_SD go high and the
//     servo get pulses (the TPS22917 ramps +5V_SW over ~40 ms).
//   * Rail off when the USB bus has been idle (no SOFs: cable pulled or host suspended) for
//     USB_IDLE_OFF_MS; it comes back with the next host frame.
// On the M5Stamp prototype PWR_EN is not connected; the policy then only delays amp/servo by 60 ms.
#pragma once
#include <stdint.h>

#include "config.h"

namespace duet {

class RailPolicy {
 public:
  // now_ms: millis(). usb_active: USB SOFs present (HWCDC::isPlugged()). host_frames: number of valid
  // host frames parsed so far (any type); a change means the host is talking.
  void tick(uint32_t now_ms, bool usb_active, uint32_t host_frames) {
    const bool new_frame = host_frames != last_frames_;
    last_frames_ = host_frames;
    if (usb_active || new_frame) last_active_ms_ = now_ms;
    if (!on_) {
      if (new_frame) {
        on_ = true;
        on_since_ms_ = now_ms;
      }
    } else if ((uint32_t)(now_ms - last_active_ms_) >= USB_IDLE_OFF_MS) {
      on_ = false;
    }
  }
  bool rail_on() const { return on_; }
  bool loads_ready(uint32_t now_ms) const {
    return on_ && (uint32_t)(now_ms - on_since_ms_) >= PWR_RAIL_SETTLE_MS;
  }

 private:
  bool on_ = false;
  uint32_t on_since_ms_ = 0;
  uint32_t last_active_ms_ = 0;
  uint32_t last_frames_ = 0;
};

}  // namespace duet
