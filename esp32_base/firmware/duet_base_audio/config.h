// Duet I/O firmware -- compile-time configuration.
// Pin map and protocol are fixed by docs/PLAN.md ("Fixed interfaces"). Change them there first.
#pragma once
#include <stdint.h>

// ---------------------------------------------------------------- pins (PLAN.md pin map)
// One build serves both boards: the custom PCB (ESP32-S3-WROOM-1-N8R2) and the M5Stamp S3/S3A
// prototype, which breaks out only G0-G15, G39-G44 and G46. That is why the amplifier sits on
// GPIO39/40/41 (2026-10-15 PLAN.md change; before: 16/17/18). GPIO39-41 are the pad-JTAG pins
// MTCK/MTDO/MTDI; with default eFuses JTAG is routed to the USB-Serial/JTAG controller (ESP32-S3
// datasheet v2.2, Table 3-5), so they are ordinary GPIOs reached through the GPIO matrix.
#define PIN_ADC_MCLK   4    // I2S0 MCLK -> PCM1808 SCKI (256 fs = 6.144 MHz)
#define PIN_ADC_BCK    5    // I2S0 BCK  -> PCM1808 BCK  (64 fs = 1.536 MHz)
#define PIN_ADC_LRCK   6    // I2S0 WS   -> PCM1808 LRCK (24 kHz)
#define PIN_ADC_DIN    7    // I2S0 DIN  <- PCM1808 DOUT
#define PIN_AMP_BCLK   15   // I2S1 BCLK -> MAX98357A BCLK (32 fs_out = 1.536 MHz)
#define PIN_AMP_LRC    39   // I2S1 WS   -> MAX98357A LRCLK (48 kHz, see below). MTCK: weak pull-up after reset
#define PIN_AMP_DOUT   40   // I2S1 DOUT -> MAX98357A DIN. MTDO: input, no pull after reset
#define PIN_AMP_SD     41   // MAX98357A SD_MODE, high = on. MTDI: input, no pull after reset, so the amp's
                            // internal 100 k pull-down holds it in shutdown until setup() drives it low
#define PIN_SERVO      13   // SG90-class servo signal (LEDC)
#define PIN_LED        14   // status LED (active high)
// PCB only: TPS22917 ON of the servo + amp rail (+5V_SW), 100 k pull-down on the board. GPIO47 floats at
// reset (datasheet Table 2-1) and is not connected on the M5Stamp S3 (M5Stack schematic v0.2), so
// driving it there has no effect. (GPIO21, its first home, is the Stamp's WS2812 data line.)
#define PIN_PWR_EN     47

// ---------------------------------------------------------------- audio
#define AUDIO_FS            24000   // wire sample rate (PLAN.md)
#define FRAME_SAMPLES       240     // 10 ms frames
// MAX98357A does NOT accept LRCLK = 24 kHz (datasheet: "LRCLK ONLY supports 8, 16, 32,
// 44.1, 48, 88.2 and 96 kHz ... 24kHz [is] NOT supported"). The speaker I2S runs at 48 kHz
// and the firmware interpolates x2 with a 63-tap half-band FIR (halfband_coefs.h).
#define SPK_FS              48000
#define SPK_UPSAMPLE        2
// Mic channel inside the PCM1808 stereo frame (0 = left / VINL, 1 = right / VINR).
#define MIC_SLOT            0

// I2S DMA sizing (see firmware/README.md "Latency budget").
//   RX: 240 frames x 2 slots x 4 B = 1920 B per descriptor (<= 4092 B limit), 4 descriptors
//       -> one DMA interrupt per 10 ms frame, 30 ms of slack before an RX overflow.
//   TX: 120 frames x 2 slots x 2 B = 480 B per descriptor = 2.5 ms at 48 kHz, 4 descriptors
//       = 10 ms of DMA capacity. With blocking writes the pipeline stays full, so the time
//       from popping a frame off the jitter buffer to its first sample leaving the pin is
//       exactly the DMA capacity (10 ms), and the speaker task may stall up to ~10 ms
//       before the DMA runs dry. (3 x 5 ms would cost 15 ms; 2 x 5 ms only 10 ms but with
//       one buffer of slack.)
#define I2S_RX_DMA_FRAMES   240
#define I2S_RX_DMA_DESC     4
#define I2S_TX_DMA_FRAMES   120
#define I2S_TX_DMA_DESC     4

// ---------------------------------------------------------------- speaker jitter buffer
// Playback starts (and restarts after an underrun) once SPK_PRIME_FRAMES 10 ms frames are queued.
// PLAN.md default: 4 frames = 40 ms. The live app measured 0 underruns with 4 frames and 2 with 2 frames
// on a heavily loaded Mac (live/README.md); predictive timing does not depend on this latency.
// Build-time configurable, e.g.:
//   arduino-cli compile ... --build-property "compiler.cpp.extra_flags=-DSPK_PRIME_FRAMES=3"
#ifndef SPK_PRIME_FRAMES
#define SPK_PRIME_FRAMES    4
#endif
#define SPK_RING_FRAMES     16                       // capacity 160 ms (power of two). Full -> newest dropped
#define SPK_TRIM_ABOVE      (SPK_PRIME_FRAMES + 2)   // if the level stays above prime + 2 frames ...
#define SPK_TRIM_AFTER      50    // ... for 50 consecutive pops (0.5 s), drop one frame
#define SPK_FADE_SAMPLES    24    // 1 ms fade-out on underrun, fade-in on resume (anti-click)
#define AMP_IDLE_OFF_MS     2000  // MAX98357A shut down after 2 s without speaker frames
static_assert(SPK_PRIME_FRAMES >= 1 && SPK_PRIME_FRAMES <= 8, "SPK_PRIME_FRAMES must be 1..8");
static_assert((SPK_RING_FRAMES & (SPK_RING_FRAMES - 1)) == 0, "SPK_RING_FRAMES must be a power of two");
static_assert(SPK_TRIM_ABOVE + 4 <= SPK_RING_FRAMES, "ring must hold the trim threshold plus a 4-frame burst");

// ---------------------------------------------------------------- servo + amp rail (PCB, PWR_EN)
// USB 2.0 (section 7.2.1.4): a high-power function first comes up below one unit load. The rail stays off
// until the first valid host frame (the host only sends after USB configuration and port open), and the
// amp/servo are driven only after the TPS22917 has ramped: t_on = 4 us/pF x CT 10 nF = 40 ms (+ margin).
#define PWR_RAIL_SETTLE_MS  60
#define USB_IDLE_OFF_MS     500   // no USB SOFs (unplugged or host suspended) this long -> rail off

// ---------------------------------------------------------------- servo
#define SERVO_HZ            50
#define SERVO_RES_BITS      14    // 20 ms / 2^14 = 1.22 us per LSB (~0.11 deg)
#define SERVO_MIN_US        500   // 0 deg
#define SERVO_MAX_US        2500  // 180 deg
#define SERVO_REST_DEG      90    // assumed start position for the first speed-limited move
// Mechanical travel limit (2026-10-16). mech/ checked the head for collisions over rest +/-25 deg
// (mech/cad/params.py NOD_UP/NOD_DOWN, check_collisions.py) and asks the firmware to keep the command within
// rest +/-20 deg, so a wrong or corrupted host value cannot drive the head into the yoke. Every SERVO command
// is clamped to SERVO_MIN_DEG..SERVO_MAX_DEG (70..110 deg); the wire format stays 0..180 (PLAN.md).
// Build-time option, e.g. -DSERVO_HALF_RANGE_DEG=15.
#ifndef SERVO_HALF_RANGE_DEG
#define SERVO_HALF_RANGE_DEG 20
#endif
#define SERVO_MIN_DEG       (SERVO_REST_DEG - SERVO_HALF_RANGE_DEG)
#define SERVO_MAX_DEG       (SERVO_REST_DEG + SERVO_HALF_RANGE_DEG)
static_assert(SERVO_HALF_RANGE_DEG >= 1 && SERVO_MIN_DEG >= 0 && SERVO_MAX_DEG <= 180,
              "servo travel must stay inside 0..180 deg");

// ---------------------------------------------------------------- timing
#define CONTROL_PERIOD_MS   10    // servo ramp / LED / amp policy tick
#define STATUS_PERIOD_MS    100   // STATUS frame period (PLAN.md)
#define HOST_TIMEOUT_MS     1000  // no valid host frame for this long -> "no host" LED pattern

// ---------------------------------------------------------------- USB CDC buffers
#define USB_RX_BUF_BYTES    4096  // ~8 SPK frames (487 B each)
#define USB_TX_BUF_BYTES    4096  // ~8 MIC frames
#define MIC_QUEUE_FRAMES    6     // audio task -> USB task hand-off (60 ms)

// ---------------------------------------------------------------- FreeRTOS placement
// Core 0: I2S ISRs + mic task + speaker task (audio never waits on USB).
// Core 1: USB-Serial/JTAG ISR (Serial.begin is called from setup() on core 1),
//         USB RX parser, USB TX writer, control task, Arduino loop (idle).
#define AUDIO_CORE          0
#define USB_CORE            1
#define PRIO_AUDIO          20    // below esp_timer (22) and IPC (24), above all other app tasks
#define PRIO_USB_RX         12
#define PRIO_USB_TX         11
#define PRIO_CONTROL        8

// Optional extra diagnostics frame (type 0x21, NOT in PLAN.md). Off by default so the wire
// format is exactly the contract. See README "Proposed protocol additions".
#ifndef DUET_SEND_DIAG
#define DUET_SEND_DIAG      0
#endif
