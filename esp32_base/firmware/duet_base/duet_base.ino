// =====================================================================================
// Duet base, sketch 1 of 2: the link and the note path. No audio, no servo, no tasks.
// Board: ESP32-S3 N16R8 dev board (16 MB flash, 8 MB octal PSRAM), native USB port.
//
//   Mac --USB (USB-Serial/JTAG, binary frames)--> DuetCore engine --> plug-in (harmony model)
//   Mac <-------------------------------------------------------------- answers
//
// Everything that decides behaviour is in the DuetCore library (firmware/libraries/DuetCore): the wire
// protocol, the engine and the plug-ins. The same files are compiled for the Mac as host_tests/board_sim,
// so this sketch only adds what is specific to the chip: the serial port, the clock, the facts in INFO and
// the on-board RGB LED.
//
// Nothing here prints text on Serial: the port carries binary frames only (text goes in LOG frames).
//
// Build (no upload):   sh tools/build_firmware.sh duet_base
// Board options used:  esp32:esp32:esp32s3  FlashSize=16M  PSRAM=opi  CDCOnBoot=cdc  USBMode=hwcdc
//                      PartitionScheme=app3M_fat9M_16MB
// =====================================================================================
#include <Arduino.h>

#include <DuetCore.h>

#include "esp_log.h"

#if !ARDUINO_USB_CDC_ON_BOOT
#error "Duet needs Serial on the native USB port: set USB CDC On Boot = Enabled (CDCOnBoot=cdc)"
#endif
#if !ARDUINO_USB_MODE
#error "Duet uses the USB-Serial/JTAG controller: set USB Mode = Hardware CDC and JTAG (USBMode=hwcdc)"
#endif

#define FW_NAME "duet_base"
#define FW_VERSION "0.1.0"

// On-board RGB LED. The generic ESP32-S3 dev module puts it on GPIO48; some board revisions use GPIO38.
// Override with  --build-property "compiler.cpp.extra_flags=-DDUET_RGB_GPIO=38". -1 = no LED.
#ifndef DUET_RGB_GPIO
#define DUET_RGB_GPIO 48
#endif

#define USB_RX_BUF_BYTES 4096
#define USB_TX_BUF_BYTES 4096

static duet::Engine g_engine;
static uint32_t g_tx_dropped = 0;    // frames not sent because the USB TX buffer was full (host not reading)
static uint32_t g_last_rx_ms = 0;    // last time bytes arrived
static bool g_host_seen = false;
static uint8_t g_led_cmd = 0xFF;     // last LED command (0 off, 1 on, 2 blink); 0xFF = none yet

static void led_rgb(uint8_t r, uint8_t g, uint8_t b) {
#if DUET_RGB_GPIO >= 0
  static uint32_t last = 0xFFFFFFFF;
  const uint32_t v = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
  if (v == last) return;  // rgbLedWrite sends a pulse train; only send it when the colour changes
  last = v;
  rgbLedWrite(DUET_RGB_GPIO, r, g, b);
#else
  (void)r; (void)g; (void)b;
#endif
}

static void led_tick() {
  const uint32_t now = millis();
  const bool host_alive = g_host_seen && (now - g_last_rx_ms < 1000);
  if (host_alive && g_led_cmd != 0xFF) {
    const bool on = g_led_cmd == 1 || (g_led_cmd == 2 && ((now / 250) & 1));  // 2 = 2 Hz blink
    led_rgb(0, on ? 24 : 0, 0);                                                 // green: the host's LED command
  } else if (host_alive) {
    led_rgb(0, 0, 12);                                                          // steady dim blue: host is talking
  } else {
    led_rgb(0, 0, (now % 1000) < 50 ? 12 : 0);                                  // blue blip each second: alive, no host
  }
}

// One whole frame or nothing: a partial frame would desynchronise the host.
static void host_write(const uint8_t* data, size_t n, void*) {
  if (Serial.availableForWrite() < (int)n) {
    ++g_tx_dropped;
    return;
  }
  Serial.write(data, n);
}

static uint32_t now_us() { return (uint32_t)micros(); }

// Facts about this chip for INFO. The first contact check reads psram and flash from here.
static size_t chip_info(char* buf, size_t cap, void*) {
  return (size_t)snprintf(buf, cap, "chip=%s;rev=%u;cpu_mhz=%u;flash=%u;psram=%u;psram_free=%u;heap_free=%u;rgb_gpio=%d;"
                                    "tx_dropped=%u;uptime_ms=%u",
                          ESP.getChipModel(), (unsigned)ESP.getChipRevision(), (unsigned)getCpuFrequencyMhz(),
                          (unsigned)ESP.getFlashChipSize(), (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreePsram(),
                          (unsigned)ESP.getFreeHeap(), (int)DUET_RGB_GPIO, (unsigned)g_tx_dropped, (unsigned)millis());
}

// Frames the engine does not answer itself.
static void on_other_frame(const duet::Frame& f, void*) {
  if (f.type == duet::T_LED) g_led_cmd = f.payload[0] <= 2 ? f.payload[0] : 0;
  // SPK_AUDIO, SERVO, AUDIO_CTRL: this sketch has no audio and no servo, so they are ignored.
}

void setup() {
  esp_log_level_set("*", ESP_LOG_NONE);  // keep the USB stream binary-only
  Serial.setRxBufferSize(USB_RX_BUF_BYTES);
  Serial.setTxBufferSize(USB_TX_BUF_BYTES);
  Serial.setTxTimeoutMs(0);  // never block on a host that is not reading
  Serial.begin();
  Serial.setDebugOutput(false);

  g_engine.begin(host_write, now_us, chip_info, nullptr, FW_NAME, FW_VERSION);
  g_engine.set_passthrough(on_other_frame, nullptr);
  led_rgb(0, 0, 0);
}

void loop() {
  static uint8_t buf[512];
  int n = Serial.available();
  if (n > 0) {
    if (n > (int)sizeof(buf)) n = sizeof(buf);
    const size_t got = Serial.read(buf, n);
    g_last_rx_ms = millis();
    g_host_seen = true;
    g_engine.feed(buf, got);
    return;  // look again at once: more bytes of the same exchange are probably on their way
  }
  g_engine.idle();
  led_tick();
  // While an exchange is going on, keep polling without sleeping (lowest delay). Otherwise sleep one tick.
  if (!g_host_seen || millis() - g_last_rx_ms > 20) delay(1);
}
