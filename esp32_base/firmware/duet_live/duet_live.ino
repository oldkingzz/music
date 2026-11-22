// =====================================================================================
// Duet live: the whole duet on the board. She sings into the microphone, the board finds her notes, the
// harmony plug-in picks Duet's note, and the board sings it on its speaker. Every note goes to the Mac.
// Board: ESP32-S3 N16R8 dev board (16 MB flash, 8 MB octal PSRAM), native USB port (USB-Serial/JTAG).
// Wiring: the team's (Christine / Xinrui), see the pin constants below. Nothing else is touched.
//
//   I2S MEMS mic --I2S RX 16 kHz--> [mic task] --hop of 256--> LiveChain (DuetCore duet_live.h):
//      (or the built-in fake melody,                  PitchTracker (the team's YIN + note decision)
//       one hop per 16 ms on the tick clock)          -> HEARD_ON / HEARD_OFF
//                                                     -> the plug-in chosen by CONFIG (Engine::plugin_step)
//                                                     -> SUNG_ON / SUNG_OFF -> synth commands
//   synth commands --lock-free ring--> [spk task] Synth (the team's wavetable) --I2S TX 22050 Hz--> amplifier
//   LIVE_NOTE / LIVE_STATE / MIC_HOP and every Engine reply --message buffer--> [USB TX task] --USB--> Mac
//   Mac --USB--> [USB RX task] --> Engine: PING, HELLO, CONFIG, NOTE_IN answered as in sketch 1;
//                                  LIVE_CTRL, SYNTH_NOTE --> LiveChain; LED --> the RGB LED
//
// Tasks and cores:
//   core 0: spk task (prio 19, paced by its blocking I2S write), mic task (prio 18, paced by its blocking I2S
//           read, or by vTaskDelayUntil for the fake melody). Both I2S drivers are started inside these tasks,
//           so their interrupts land on core 0 and each I2S object is used by one task only.
//   core 1: USB RX task (prio 12), USB TX task (prio 11, the only task that writes to Serial, whole frames
//           only), Arduino loop (LED, every 10 ms).
// One lock (g_lock, a FreeRTOS mutex) serialises every call into the Engine and the LiveChain: the USB RX task
// holds it around Engine::feed / idle (CONFIG swaps the plug-in, LIVE_CTRL changes the chain), the mic task
// around each hop (the plug-in call, the frames it sends), the audio tasks around their error LOG. Nothing
// waits for anything else while holding it. The synth commands cross from the lock holder to the speaker task
// through LiveVoice::q, a single-producer single-consumer ring (single producer because of the lock).
//
// NOTHING IS DRIVEN AT BOOT. No I2S driver is installed and no audio pin is configured until the host asks with
// LIVE_CTRL: the microphone I2S starts only when LISTEN is on with the real microphone selected (LC_FAKE clear)
// and stops when that ends; the speaker I2S starts only when SING is on and stops (after the team's short fade)
// when SING goes off. With LC_FAKE the chain runs on a melody generated on the board (DuetCore duet_fake16.h;
// host/duet_host/fake_audio.py makes the same samples), so a bare board with nothing wired runs the whole loop.
// The answer to LIVE_CTRL shows the gates at once; the I2S drivers come up a moment later in their tasks
// (LS_MIC_UP / LS_SPK_UP). The host sends the same LIVE_CTRL again to see them (link.live_ctrl does that).
// If an I2S begin fails: LS_ERROR is set, a LOG frame says which one, the link keeps running; the next request
// (the gate off and on again) retries.
//
// Nothing here prints text on Serial: the port carries binary frames only (text goes in LOG frames).
//
// Credits: the pitch tracker is by Christine (GitHub cchristinem: quantization_code_3, branch Christine of
// github.com/oldkingzz/music, commit bde720a); the synth is by Xinrui Gao (GitHub XinruiGao: quantization_code_4,
// branch SIgrid, commit ff33e2d). Both are lifted into DuetCore unchanged in their constants and arithmetic
// (duet_pitch.h, duet_synth.h list the mechanical changes); the I2S settings below are theirs.
//
// Build (no upload):   sh tools/build_firmware.sh duet_live
// Upload (only after the owner says so):  sh tools/upload_firmware.sh duet_live --yes-upload
// Check (simulator):   .venv/bin/python host/checks/check_live.py --target sim-live
// Board options used:  esp32:esp32:esp32s3  FlashSize=16M  PSRAM=opi  CDCOnBoot=cdc  USBMode=hwcdc
//                      PartitionScheme=app3M_fat9M_16MB
// =====================================================================================
#include <Arduino.h>
#include <ESP_I2S.h>

#include <DuetCore.h>

#include <atomic>

#include "freertos/message_buffer.h"

#if !ARDUINO_USB_CDC_ON_BOOT
#error "Duet needs Serial on the native USB port: set USB CDC On Boot = Enabled (CDCOnBoot=cdc)"
#endif
#if !ARDUINO_USB_MODE
#error "Duet uses the USB-Serial/JTAG controller: set USB Mode = Hardware CDC and JTAG (USBMode=hwcdc)"
#endif
#if ((configTICK_RATE_HZ * 16) % 1000) != 0
#error "the fake microphone needs 16 ms (one hop) to be a whole number of FreeRTOS ticks"
#endif

#define FW_NAME "duet_live"
#define FW_VERSION "0.3.0"

// ------------------------------------------------------------------ the team's wiring (the only pins used)
// Microphone: I2S MEMS, L/R tied to GND = left slot.
static const int MIC_PIN_BCLK = 11;  // BCLK / SCK
static const int MIC_PIN_WS = 12;    // WS
static const int MIC_PIN_SD = 13;    // SD, data into the ESP32
// Speaker amplifier: I2S (MAX98357A or similar).
static const int SPK_PIN_BCLK = 4;   // BCLK
static const int SPK_PIN_LRC = 5;    // LRC / WS
static const int SPK_PIN_DIN = 6;    // DIN, data out of the ESP32
static const uint32_t MIC_RATE = 16000;  // the pitch tracker's SAMPLE_RATE
static const uint32_t SPK_RATE = 22050;  // the synth's PLAY_RATE

// On-board RGB LED, as in sketch 1. GPIO48 on the generic ESP32-S3 dev module; -1 = no LED.
#ifndef DUET_RGB_GPIO
#define DUET_RGB_GPIO 48
#endif

using namespace duet;

static_assert(MIC_RATE == PT_SAMPLE_RATE, "the microphone runs at the tracker's rate");
static_assert(SPK_RATE == SY_PLAY_RATE, "the speaker runs at the synth's rate");
static_assert(LIVE_HOP_MS == 16, "the #error guard above assumes a 16 ms hop");

// ------------------------------------------------------------------ sizes, priorities
#define USB_RX_BUF_BYTES 4096
#define USB_TX_BUF_BYTES 8192
#define TX_MB_BYTES 16384  // frames waiting for the USB TX task (about 0.4 s of MIC_HOP + events)
#define TX_FRAME_MAX (HEADER_LEN + MAX_DEVICE_PAYLOAD + 1)  // 519 bytes (MIC_HOP)
#define TX_ROOM_WAIT_MS 10  // the TX task waits this long for room in the USB buffer before it drops a frame
#define AUDIO_CORE 0
#define USB_CORE 1
#define PRIO_SPK 19
#define PRIO_MIC 18
#define PRIO_USB_RX 12
#define PRIO_USB_TX 11
#define TASK_STACK_MIC 8192
#define TASK_STACK_SPK 8192  // I2SClass::begin / end run here (the team gives that task 8192)
#define TASK_STACK_RX 8192
#define TASK_STACK_TX 4096

// ------------------------------------------------------------------ shared state
static Engine g_engine;    // under g_lock
static LiveChain g_chain;  // under g_lock (its gate getters are atomic)
static LiveVoice g_voice;  // speaker task; its ring is filled by the g_lock holder
static I2SClass g_mic;     // mic task only
static I2SClass g_spk;     // spk task only
static SemaphoreHandle_t g_lock;
static MessageBufferHandle_t g_tx_mb;
static TaskHandle_t g_mic_task = nullptr, g_spk_task = nullptr;

static std::atomic<bool> g_mic_up{false}, g_spk_up{false};    // I2S running (written by its own task)
static std::atomic<bool> g_mic_err{false}, g_spk_err{false};  // the last begin failed (cleared when the gate goes off)
static std::atomic<uint32_t> g_tx_dropped{0};   // frames not sent: message buffer full or USB buffer full
static std::atomic<uint32_t> g_mic_short{0};    // microphone reads that came back short (I2S error / timeout)
static std::atomic<uint32_t> g_spk_short{0};    // speaker writes that came back short
static std::atomic<uint32_t> g_last_rx_ms{0};
static std::atomic<bool> g_host_seen{false};
static std::atomic<uint8_t> g_led_cmd{0xFF};    // last LED command (0 off, 1 on, 2 blink); 0xFF = none yet

static inline void lock() { xSemaphoreTake(g_lock, portMAX_DELAY); }
static inline void unlock() { xSemaphoreGive(g_lock); }

// The engine's writer (every frame the device sends goes through it, always under g_lock): one whole frame
// into the message buffer, or nothing. Never waits.
static void engine_write(const uint8_t* data, size_t n, void*) {
  if (xMessageBufferSend(g_tx_mb, data, n, 0) != n) g_tx_dropped++;
}

static uint32_t now_us() { return (uint32_t)micros(); }

// INFO: the engine puts fw, ver, proto, plugins first; live_info appends whole fields in order of need
// (flash, psram, the live keys, then sketch 1's chip facts) and leaves out what does not fit in 240 bytes.
static size_t chip_info(char* buf, size_t cap, void*) {
  LiveInfoFacts f;
  f.chip = ESP.getChipModel();
  f.rev = (unsigned)ESP.getChipRevision();
  f.cpu_mhz = (unsigned)getCpuFrequencyMhz();
  f.flash = (uint32_t)ESP.getFlashChipSize();
  f.psram = (uint32_t)ESP.getPsramSize();
  f.psram_free = (uint32_t)ESP.getFreePsram();
  f.heap_free = (uint32_t)ESP.getFreeHeap();
  f.rgb_gpio = DUET_RGB_GPIO;
  f.tx_dropped = g_tx_dropped.load();
  f.uptime_ms = (uint32_t)millis();
  f.mic_rate = MIC_RATE;
  f.spk_rate = SPK_RATE;
  f.mic_pins[0] = MIC_PIN_BCLK;
  f.mic_pins[1] = MIC_PIN_WS;
  f.mic_pins[2] = MIC_PIN_SD;
  f.spk_pins[0] = SPK_PIN_BCLK;
  f.spk_pins[1] = SPK_PIN_LRC;
  f.spk_pins[2] = SPK_PIN_DIN;
  return live_info(buf, cap, f);
}

// Hardware facts for LIVE_STATE.
static void live_hw(LiveHw* hw, void*) {
  hw->mic_up = g_mic_up.load();
  hw->spk_up = g_spk_up.load();
  hw->error = g_mic_err.load() || g_spk_err.load();
}

// A LOG frame from an audio task (not holding the lock).
static void log_text(const char* text) {
  lock();
  g_engine.log(text);
  unlock();
}

// Frames the engine does not answer itself. Runs in the USB RX task, under g_lock.
static void on_other_frame(const Frame& f, void*) {
  if (g_chain.on_frame(f)) {  // LIVE_CTRL, SYNTH_NOTE
    if (g_mic_task) xTaskNotifyGive(g_mic_task);  // react to the new gates now, not at the next timeout
    if (g_spk_task) xTaskNotifyGive(g_spk_task);
    return;
  }
  if (f.type == T_LED) g_led_cmd.store(f.payload[0] <= 2 ? f.payload[0] : 0);
  // MIC_AUDIO, SPK_AUDIO, SERVO, AUDIO_CTRL, AUDIO_QUERY: not part of this sketch, ignored.
}

// ------------------------------------------------------------------ core 0: microphone
static bool mic_begin() {
  g_mic.setPins(MIC_PIN_BCLK, MIC_PIN_WS, -1, MIC_PIN_SD);  // the team's: i2s.setPins(11, 12, -1, 13)
  return g_mic.begin(I2S_MODE_STD, MIC_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT);
}

// Microphone task: the source of the chain. Real microphone: blocking reads of one hop (256 32-bit words), as
// the team's audioTask does. Fake melody: one hop every 16 ms on an absolute tick schedule (vTaskDelayUntil,
// restarted at every restart of the source). Each hop goes through the chain under g_lock.
static void mic_task(void*) {
  static int32_t raw[PT_HOP];
  bool fake_running = false;
  uint32_t epoch_seen = 0;
  TickType_t wake = 0;
  for (;;) {
    const bool want_mic = g_chain.want_mic();
    // the real microphone's I2S follows its gate
    if (want_mic && !g_mic_up.load() && !g_mic_err.load()) {
      if (mic_begin()) {
        g_mic_up.store(true);
      } else {
        g_mic.end();
        g_mic_err.store(true);
        log_text("duet_live: microphone I2S begin failed (BCLK 11, WS 12, SD 13, 16 kHz, 32-bit, left slot)");
      }
    } else if (!want_mic && g_mic_up.load()) {
      g_mic.end();
      g_mic_up.store(false);
    }
    if (!want_mic) g_mic_err.store(false);

    if (want_mic && g_mic_up.load()) {
      fake_running = false;
      const size_t got = g_mic.readBytes((char*)raw, sizeof(raw));  // blocks about 16 ms
      if (got < sizeof(raw)) {  // their: continue. Here also a pause, so an I2S error cannot spin the core.
        g_mic_short++;
        vTaskDelay(1);
        continue;
      }
      lock();
      if (g_chain.want_mic()) g_chain.hop_raw(raw);
      unlock();
      continue;
    }
    if (g_chain.listening() && g_chain.fake()) {
      if (!fake_running || g_chain.epoch() != epoch_seen) {
        fake_running = true;
        epoch_seen = g_chain.epoch();
        wake = xTaskGetTickCount();
      }
      vTaskDelayUntil(&wake, pdMS_TO_TICKS(LIVE_HOP_MS));
      lock();
      if (g_chain.listening() && g_chain.fake() && g_chain.epoch() == epoch_seen) g_chain.hop_fake();
      unlock();
      continue;
    }
    fake_running = false;
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20));  // nothing to listen to: wait for LIVE_CTRL
  }
}

// ------------------------------------------------------------------ core 0: speaker
static bool spk_begin() {
  g_spk.setPins(SPK_PIN_BCLK, SPK_PIN_LRC, SPK_PIN_DIN);  // the team's: spk.setPins(4, 5, 6)
  return g_spk.begin(I2S_MODE_STD, SPK_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO, I2S_STD_SLOT_BOTH);
}

// their writeBlock: 16-bit stereo, 4 bytes per frame, the whole block or false
static bool write_block(const int16_t* buf, size_t frames) {
  const uint8_t* b = (const uint8_t*)buf;
  size_t total = frames * 4, off = 0;
  while (off < total) {
    size_t n = g_spk.write(b + off, total - off);
    if (n == 0) return false;
    off += n;
  }
  return true;
}

// Speaker task: while SING is on, renders SY_PLAY_BLOCK frames of the synth and writes them; the blocking write
// paces it at 22050 Hz. With SING off the speaker I2S is not running and nothing is rendered.
static void spk_task(void*) {
  static int16_t lr[2 * SY_PLAY_BLOCK];
  for (;;) {
    const bool want = g_chain.singing();
    const bool up = g_spk_up.load();
    if (want && !up) {
      if (!g_spk_err.load()) {
        if (spk_begin()) {
          g_voice.start();
          memset(lr, 0, sizeof(lr));
          write_block(lr, SY_PLAY_BLOCK);  // their "a little silence first"
          write_block(lr, SY_PLAY_BLOCK);
          g_spk_up.store(true);
        } else {
          g_spk.end();
          g_spk_err.store(true);
          log_text("duet_live: speaker I2S begin failed (BCLK 4, LRC 5, DIN 6, 22050 Hz, 16-bit stereo)");
        }
      } else {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20));
      }
      continue;
    }
    if (want && up) {
      g_voice.block(lr, g_chain.gain());
      if (!write_block(lr, SY_PLAY_BLOCK)) {  // an I2S error: count it and pause, so it cannot spin the core
        g_spk_short++;
        vTaskDelay(1);
      }
      continue;
    }
    if (!want && up) {  // their stop: quick fade so stopping does not click, trailing silence, release the port
      g_voice.begin_stop();
      while (g_voice.stop_block(lr, g_chain.gain())) write_block(lr, SY_PLAY_BLOCK);
      memset(lr, 0, sizeof(lr));
      write_block(lr, SY_PLAY_BLOCK);
      write_block(lr, SY_PLAY_BLOCK);
      g_spk.end();
      g_spk_up.store(false);
      continue;
    }
    g_spk_err.store(false);  // SING is off: the next SING retries a failed begin
    g_voice.drop_pending();
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20));
  }
}

// ------------------------------------------------------------------ core 1: USB
// USB RX task: reads the port and feeds the engine under the lock. Sleeps 1 tick when there is nothing to read
// (or it would starve the TX task on the same core).
static void usb_rx_task(void*) {
  static uint8_t buf[512];
  for (;;) {
    int n = Serial.available();
    if (n > 0) {
      if (n > (int)sizeof(buf)) n = sizeof(buf);
      const size_t got = Serial.read(buf, n);
      g_last_rx_ms.store(millis());
      g_host_seen.store(true);
      lock();
      g_engine.feed(buf, got);
      unlock();
    } else {
      lock();
      g_engine.idle();
      unlock();
      vTaskDelay(1);
    }
  }
}

// USB TX task: the only place that writes to Serial. Whole frame or nothing: a partial frame would desynchronise
// the host. Waits up to TX_ROOM_WAIT_MS for room in the USB buffer, then drops the frame (host not reading).
static void usb_tx_task(void*) {
  static uint8_t out[TX_FRAME_MAX];
  for (;;) {
    const size_t n = xMessageBufferReceive(g_tx_mb, out, sizeof(out), pdMS_TO_TICKS(20));
    if (!n) continue;
    int waited = 0;
    while (Serial.availableForWrite() < (int)n && waited < TX_ROOM_WAIT_MS) {
      vTaskDelay(1);
      ++waited;
    }
    if (Serial.availableForWrite() < (int)n || Serial.write(out, n) != n) g_tx_dropped++;
  }
}

// ------------------------------------------------------------------ the RGB LED (Arduino loop task)
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

// As sketch 1, plus red for an I2S error and a dim white while the chain listens.
static void led_tick() {
  const uint32_t now = millis();
  const bool host_alive = g_host_seen.load() && (now - g_last_rx_ms.load() < 1000);
  const uint8_t cmd = g_led_cmd.load();
  if (g_mic_err.load() || g_spk_err.load()) {
    led_rgb(((now / 100) & 1) ? 24 : 0, 0, 0);                                  // red 5 Hz: an I2S begin failed
  } else if (host_alive && cmd != 0xFF) {
    const bool on = cmd == 1 || (cmd == 2 && ((now / 250) & 1));
    led_rgb(0, on ? 24 : 0, 0);                                                 // green: the host's LED command
  } else if (g_chain.listening()) {
    led_rgb(8, 8, 8);                                                           // dim white: listening
  } else if (host_alive) {
    led_rgb(0, 0, 12);                                                          // steady dim blue: host is talking
  } else {
    led_rgb(0, 0, (now % 1000) < 50 ? 12 : 0);                                  // blue blip each second: no host
  }
}

// ------------------------------------------------------------------ Arduino entry points
void setup() {
  esp_log_level_set("*", ESP_LOG_NONE);  // keep the USB stream binary-only
  Serial.setRxBufferSize(USB_RX_BUF_BYTES);
  Serial.setTxBufferSize(USB_TX_BUF_BYTES);
  Serial.setTxTimeoutMs(0);  // never block a task on a host that is not reading
  Serial.begin();
  Serial.setDebugOutput(false);

  g_lock = xSemaphoreCreateMutex();
  g_tx_mb = xMessageBufferCreate(TX_MB_BYTES);
  g_engine.begin(engine_write, now_us, chip_info, nullptr, FW_NAME, FW_VERSION);
  g_engine.set_passthrough(on_other_frame, nullptr);
  g_chain.begin(&g_engine, &g_voice, live_hw, nullptr);
  led_rgb(0, 0, 0);

  // Order matters: the audio task handles exist before the RX task can notify them. Neither audio task starts
  // an I2S driver until LIVE_CTRL asks for it.
  xTaskCreatePinnedToCore(spk_task, "spk", TASK_STACK_SPK, nullptr, PRIO_SPK, &g_spk_task, AUDIO_CORE);
  xTaskCreatePinnedToCore(mic_task, "mic", TASK_STACK_MIC, nullptr, PRIO_MIC, &g_mic_task, AUDIO_CORE);
  xTaskCreatePinnedToCore(usb_tx_task, "usb_tx", TASK_STACK_TX, nullptr, PRIO_USB_TX, nullptr, USB_CORE);
  xTaskCreatePinnedToCore(usb_rx_task, "usb_rx", TASK_STACK_RX, nullptr, PRIO_USB_RX, nullptr, USB_CORE);
}

void loop() {
  led_tick();
  vTaskDelay(pdMS_TO_TICKS(10));
}
