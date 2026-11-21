// =====================================================================================
// Duet base, sketch 2 of 2: the link and the note path of sketch 1, plus audio, servo, LED and power policy.
// Board: ESP32-S3 N16R8 dev board (16 MB flash, 8 MB octal PSRAM), native USB port (USB-Serial/JTAG).
//
//   Mac --USB--> [USB RX task] --> DuetCore engine: PING, HELLO, CONFIG, NOTE_IN answered here
//                                    |  every other frame goes to on_other_frame():
//                                    |    SPK_AUDIO  --jitter buffer--> [spk task] --x2 FIR--> I2S1 48 kHz --> MAX98357A
//                                    |    SERVO, LED --ctrl queue--> [control task 100 Hz] --> servo, LEDs, AMP_SD, PWR_EN
//                                    |    AUDIO_CTRL --> starts audio (once), sets the mic / speaker gates
//   PCM1808 --I2S0 RX 24 kHz--> [mic task] --mic queue--> [USB TX task] --USB--> Mac
//   engine replies, STATUS ----------------- reply queue --> [USB TX task]
//
// Core 0: audio only (I2S ISRs, mic task, speaker task), so audio never waits on USB.
// Core 1: USB-Serial/JTAG ISR, USB RX task, USB TX task, control task, Arduino loop (idle).
//
// AUDIO IS OFF AT BOOT. Until the host sends AUDIO_CTRL with bit0 (send MIC_AUDIO) or bit1 (play SPK_AUDIO) set,
// no I2S driver is installed, no audio pin is clocked, no MIC_AUDIO or STATUS frame is sent and the amplifier
// shutdown pin is held low. So a board with nothing wired is fully usable for the link and the note path.
// The power rail (PWR_EN) and the servo keep the old duet_io rules (power_policy.h, servo_ramp.h): the rail
// goes on at the first valid host frame of any type, the servo gets pulses only after a SERVO command.
//
// Single writer: only the USB TX task calls Serial.write, and it writes whole frames or nothing.
// Nothing here prints text on Serial: the port carries binary frames only.
//
// The files copied from duet_v0/firmware/duet_io (audio_io, jitter_buffer, upsampler, halfband_coefs,
// servo_ramp, power_policy, config) are unchanged; the pin map is in config.h. To move a pin, edit config.h:
// audio_io.cpp is compiled on its own and reads the pins from there.
//
// Build (no upload):   sh tools/build_firmware.sh duet_base_audio
// =====================================================================================
#include <Arduino.h>

#include <DuetCore.h>

#include <atomic>

#include "audio_io.h"
#include "config.h"
#include "esp_log.h"
#include "jitter_buffer.h"
#include "power_policy.h"
#include "servo_ramp.h"
#include "upsampler.h"

#if !ARDUINO_USB_CDC_ON_BOOT
#error "Duet needs Serial on the native USB port: set USB CDC On Boot = Enabled (CDCOnBoot=cdc)"
#endif
#if !ARDUINO_USB_MODE
#error "Duet uses the USB-Serial/JTAG controller: set USB Mode = Hardware CDC and JTAG (USBMode=hwcdc)"
#endif

#define FW_NAME "duet_base_audio"
#define FW_VERSION "0.1.0"

// On-board RGB LED, as in sketch 1. GPIO48 on the generic ESP32-S3 dev module; -1 = no LED.
#ifndef DUET_RGB_GPIO
#define DUET_RGB_GPIO 48
#endif

using namespace duet;

// ------------------------------------------------------------------ sizes
// The engine sends at most a 240-byte payload (INFO, LOG); MIC_AUDIO (480 bytes) is never sent through it.
static const size_t REPLY_MAX = HEADER_LEN + MAX_TEXT + 1;  // 247 bytes
static const size_t MIC_FRAME_BYTES = HEADER_LEN + FRAME_SAMPLES * 2 + 1;  // 487 bytes
#define REPLY_QUEUE_DEPTH 16
#define REPLY_WAIT_MS 5      // the engine's writer may wait this long for room in the reply queue
#define TASK_STACK_RX 8192   // engine: INFO text + snprintf, a 480-byte SPK copy
#define TASK_STACK_TX 4096   // frame buffers are static
#define TASK_STACK_CTRL 8192 // first rgbLedWrite / ledcAttach calls
#define TASK_STACK_AUDIO 4096
#define TASK_STACK_SPK 4096

// AUDIO_CTRL bits
#define AC_MIC 0x01
#define AC_SPK 0x02

// ------------------------------------------------------------------ shared state
static Engine g_engine;  // used by the USB RX task only
static AudioIO g_audio;  // filled by audio_begin on core 0, then used by the mic and speaker tasks
static JitterBuffer g_jb;  // single producer (USB RX task) / single consumer (speaker task)
static Upsampler2x g_up;   // speaker task only

struct MicItem {  // mic task -> USB TX task
  uint8_t seq;
  int16_t pcm[FRAME_SAMPLES];
};
struct CtrlCmd {  // USB RX task -> control task
  uint8_t type, a, b;
};
struct TxFrame {  // a finished, encoded frame: engine replies (RX task) and STATUS/DIAG (control task) -> USB TX
  uint16_t n;
  uint8_t bytes[REPLY_MAX];
};

static QueueHandle_t g_mic_q;    // MicItem
static QueueHandle_t g_reply_q;  // TxFrame
static QueueHandle_t g_ctrl_q;   // CtrlCmd
static TaskHandle_t g_usb_tx_task = nullptr;
static TaskHandle_t g_audio_task = nullptr;

// Counters (u32, reported mod 2^16 in STATUS). Each is atomic or written from one place only.
static volatile uint32_t g_mic_dma_ovf = 0;      // I2S RX DMA queue overflow (ISR only)
static std::atomic<uint32_t> g_mic_q_drop{0};    // mic queue full (USB TX task starved)
static std::atomic<uint32_t> g_mic_usb_drop{0};  // USB TX buffer full when a mic frame was due
static std::atomic<uint32_t> g_mic_read_err{0};  // i2s_channel_read timeout/error
static std::atomic<uint32_t> g_spk_seq_gaps{0};  // SPK_AUDIO seq discontinuities
static std::atomic<uint32_t> g_tx_dropped{0};    // replies/STATUS not sent: reply queue or USB TX buffer full
static std::atomic<uint32_t> g_last_host_ms{0};  // last valid host frame (any type)
static std::atomic<uint32_t> g_host_frames{0};   // valid host frames so far (drives the PWR_EN policy)
static std::atomic<uint32_t> g_last_spk_ms{0};   // last accepted SPK_AUDIO frame
static std::atomic<bool> g_host_seen{false};
static std::atomic<bool> g_spk_seen{false};
// Audio state. g_mic_on / g_spk_on are written by the USB RX task only (AUDIO_CTRL).
// g_audio_ready / g_audio_err are written by the audio task only.
static std::atomic<bool> g_mic_on{false};
static std::atomic<bool> g_spk_on{false};
static std::atomic<bool> g_audio_ready{false};  // I2S installed and running
static std::atomic<int> g_audio_err{0};         // first audio_begin error (esp_err_t), 0 = none

static inline uint32_t mic_overruns() {
  return g_mic_dma_ovf + g_mic_q_drop.load() + g_mic_usb_drop.load() + g_mic_read_err.load();
}

static inline void kick_tx() { if (g_usb_tx_task) xTaskNotifyGive(g_usb_tx_task); }

// Encode a small frame (STATUS, DIAG) into its own buffer and queue it for the USB TX task.
// Called from the control task. Never waits: if the queue is full the frame is dropped and counted.
static void post_frame(uint8_t type, uint8_t seq, const uint8_t* p, uint16_t len) {
  static TxFrame t;  // control task only
  if ((size_t)HEADER_LEN + len + 1 > REPLY_MAX) return;
  t.n = (uint16_t)encode(type, seq, p, len, t.bytes);
  if (xQueueSend(g_reply_q, &t, 0) != pdTRUE) g_tx_dropped++;
  kick_tx();
}

// ------------------------------------------------------------------ core 0: audio
// Speaker task (core 0, created by the audio task after I2S is up): pops one 10 ms frame (audio or faded
// silence), upsamples x2 and writes it to I2S1. The blocking write is what paces it at exactly 48 kHz.
static void spk_task(void*) {
  static int16_t frame[FRAME_SAMPLES];
  static int16_t lr[FRAME_SAMPLES * SPK_UPSAMPLE * 2];
  for (;;) {
    g_jb.pop(frame);
    g_up.process_stereo(frame, FRAME_SAMPLES, lr);
    audio_write_spk(g_audio, lr, FRAME_SAMPLES * SPK_UPSAMPLE, 100);
  }
}

// Audio task (core 0). Created at boot, but sleeps until the USB RX task wakes it with the first AUDIO_CTRL
// that asks for audio. Only then it installs the I2S drivers (here, so their interrupts land on core 0),
// starts the speaker task and becomes the mic reader. It reads the microphone all the time once started, so
// the DMA never overflows; frames are queued for the host only while the mic gate (AUDIO_CTRL bit0) is on.
static void audio_task(void*) {
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);  // nothing touches I2S before this

  const int err = audio_begin(g_audio, &g_mic_dma_ovf);
  g_audio_err.store(err);
  if (err != 0) {
    vTaskDelete(nullptr);  // control task shows the error on the LEDs; USB keeps running; no retry
    return;
  }
  g_audio_ready.store(true);
  // Speaker task on the same core, same priority; both are DMA-paced and mostly blocked.
  xTaskCreatePinnedToCore(spk_task, "spk", TASK_STACK_SPK, nullptr, PRIO_AUDIO, nullptr, AUDIO_CORE);

  static MicItem item;
  uint8_t seq = 0;
  for (;;) {
    if (!audio_read_mic(g_audio, item.pcm, 100)) {
      g_mic_read_err++;
      continue;
    }
    if (!g_mic_on.load()) continue;  // gate closed: read and discard
    item.seq = seq++;  // assigned to queued frames only -> every lost frame shows as a seq gap on the host
    if (xQueueSend(g_mic_q, &item, 0) != pdTRUE) g_mic_q_drop++;
    kick_tx();
  }
}

// ------------------------------------------------------------------ core 1: USB
// The engine's writer. Runs in the USB RX task (the engine runs there): copies the finished frame into the
// reply queue. It may wait a few ms for room, because the RX task outranks the TX task and a single read can
// hold dozens of PINGs; the TX task never blocks on USB, so this wait always ends.
static void engine_write(const uint8_t* data, size_t n, void*) {
  static TxFrame t;  // USB RX task only
  if (n > REPLY_MAX) {
    g_tx_dropped++;
    return;
  }
  t.n = (uint16_t)n;
  memcpy(t.bytes, data, n);
  if (xQueueSend(g_reply_q, &t, pdMS_TO_TICKS(REPLY_WAIT_MS)) != pdTRUE) g_tx_dropped++;
  kick_tx();
}

static uint32_t now_us() { return (uint32_t)micros(); }

// INFO facts (the engine adds fw, ver, proto, plugins in front, about 62 bytes; the whole payload is capped
// at 240). flash and psram come first because the host's first contact check reads them. Optional fields
// are appended only if the whole key=value fits, so nothing is ever cut in half.
static size_t append(char* buf, size_t cap, size_t n, const char* text) {
  const size_t k = strlen(text);
  if (n + k + 1 > cap) return n;  // +1 for the NUL
  memcpy(buf + n, text, k + 1);
  return n + k;
}

static size_t chip_info(char* buf, size_t cap, void*) {
  if (cap == 0) return 0;
  char tmp[48];
  buf[0] = 0;
  size_t n = 0;
  snprintf(tmp, sizeof(tmp), "flash=%u;psram=%u", (unsigned)ESP.getFlashChipSize(), (unsigned)ESP.getPsramSize());
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";audio=%d;audio_err=%d;mic=%d;spk=%d", g_audio_ready.load() ? 1 : 0,
           g_audio_err.load(), g_mic_on.load() ? 1 : 0, g_spk_on.load() ? 1 : 0);
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";chip=%s;rev=%u;cpu_mhz=%u", ESP.getChipModel(), (unsigned)ESP.getChipRevision(),
           (unsigned)getCpuFrequencyMhz());
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";heap_free=%u;rgb_gpio=%d", (unsigned)ESP.getFreeHeap(), (int)DUET_RGB_GPIO);
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";tx_dropped=%u", (unsigned)g_tx_dropped.load());
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";psram_free=%u", (unsigned)ESP.getFreePsram());
  n = append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";uptime_ms=%u", (unsigned)millis());
  n = append(buf, cap, n, tmp);
  return n;  // <= cap - 1
}

// Frames the engine does not answer itself. Runs in the USB RX task.
static bool s_have_spk_seq = false;  // USB RX task only
static uint8_t s_last_spk_seq = 0;   // USB RX task only
static bool s_audio_started = false; // USB RX task only: one-shot latch, never retried

static void on_other_frame(const Frame& f, void*) {
  switch (f.type) {
    case T_SPK_AUDIO: {
      // Accepted only while the speaker gate is on and I2S is running; otherwise dropped.
      if (!g_spk_on.load() || !g_audio_ready.load()) break;
      if (s_have_spk_seq && (uint8_t)(s_last_spk_seq + 1) != f.seq) g_spk_seq_gaps++;
      s_have_spk_seq = true;
      s_last_spk_seq = f.seq;
      int16_t pcm[FRAME_SAMPLES];
      memcpy(pcm, f.payload, sizeof(pcm));  // little-endian on both ends
      g_jb.push(pcm);
      g_last_spk_ms.store(millis());
      g_spk_seen.store(true);
      break;
    }
    case T_SERVO:
    case T_LED: {
      CtrlCmd c{f.type, f.payload[0], (uint8_t)(f.len > 1 ? f.payload[1] : 0)};
      xQueueSend(g_ctrl_q, &c, 0);
      break;
    }
    case T_AUDIO_CTRL: {
      const uint8_t flags = f.payload[0];
      const bool spk = (flags & AC_SPK) != 0;
      if (spk && !g_spk_on.load()) s_have_spk_seq = false;  // new stream: do not count the pause as a gap
      g_mic_on.store((flags & AC_MIC) != 0);
      g_spk_on.store(spk);
      if ((flags & (AC_MIC | AC_SPK)) && !s_audio_started) {
        s_audio_started = true;
        xTaskNotifyGive(g_audio_task);  // audio_begin runs on core 0, in the audio task
      }
      break;
    }
    default:
      break;  // dev->host types arriving here are ignored
  }
}

// USB RX task (core 1, highest app priority on this core): reads the port and feeds the engine. It must sleep
// (vTaskDelay) when there is nothing to read, or it would starve the TX and control tasks on the same core;
// this adds at most 1 ms to host->device latency. Every valid frame of any type (the engine's parser counts
// them) marks the host as alive and drives the PWR_EN policy.
static void usb_rx_task(void*) {
  static uint8_t buf[512];
  uint32_t seen_ok = 0;
  for (;;) {
    int n = Serial.available();
    if (n > 0) {
      if (n > (int)sizeof(buf)) n = sizeof(buf);
      const size_t got = Serial.read(buf, n);
      g_engine.feed(buf, got);
    } else {
      g_engine.idle();
    }
    const uint32_t ok = g_engine.parser_stats().frames_ok;
    if (ok != seen_ok) {
      seen_ok = ok;
      g_last_host_ms.store(millis());
      g_host_seen.store(true);
      g_host_frames.store(ok);
    }
    if (n <= 0) vTaskDelay(1);
  }
}

// The only place that writes to Serial. Whole frame or nothing: a partial frame would desynchronise the host.
static bool write_frame(const uint8_t* data, size_t n) {
  if (Serial.availableForWrite() < (int)n) return false;
  return Serial.write(data, n) == n;
}

static void send_replies() {
  static TxFrame t;  // USB TX task only
  while (xQueueReceive(g_reply_q, &t, 0) == pdTRUE) {
    if (!write_frame(t.bytes, t.n)) g_tx_dropped++;
  }
}

// USB TX task (core 1): wakes when something is queued (or every 20 ms), sends replies first, then mic frames,
// and sends any new reply between two mic frames, so control replies never wait behind a backlog of audio.
static void usb_tx_task(void*) {
  static MicItem item;
  static uint8_t out[MIC_FRAME_BYTES];
  for (;;) {
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20));
    send_replies();
    while (xQueueReceive(g_mic_q, &item, 0) == pdTRUE) {
      if (g_mic_on.load()) {  // frames queued just before the gate closed are discarded
        const size_t n = encode(T_MIC_AUDIO, item.seq, (const uint8_t*)item.pcm, FRAME_SAMPLES * 2, out);
        if (!write_frame(out, n)) g_mic_usb_drop++;
      }
      send_replies();
    }
  }
}

// ------------------------------------------------------------------ core 1: control
static inline bool usb_bus_active() {
  return HWCDC::isPlugged();  // USB SOFs seen recently (ESP-IDF usb_serial_jtag_is_connected)
}

static void led_rgb(uint8_t r, uint8_t g, uint8_t b) {
#if DUET_RGB_GPIO >= 0
  static uint32_t last = 0xFFFFFFFF;  // control task only
  const uint32_t v = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
  if (v == last) return;  // rgbLedWrite sends a pulse train; only send it when the colour changes
  last = v;
  rgbLedWrite(DUET_RGB_GPIO, r, g, b);
#else
  (void)r; (void)g; (void)b;
#endif
}

// Control task (core 1, every 10 ms): servo ramp, power rail, GPIO14 LED, on-board RGB LED, amplifier enable,
// STATUS. The only task that touches these pins, so none of them needs a lock.
static void control_task(void*) {
  ServoRamp servo;
  RailPolicy rail;
  bool rail_pin = false;
  bool servo_attached = false;
  uint32_t last_duty = 0xFFFFFFFF;
  uint8_t led_state = 0;
  bool led_cmd_seen = false;
  bool amp_on = false;
  uint8_t seq_status = 0, seq_diag = 0;
  uint32_t tick = 0;
  TickType_t wake = xTaskGetTickCount();

  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(CONTROL_PERIOD_MS));
    ++tick;
    const uint32_t now = millis();

    CtrlCmd c;
    while (xQueueReceive(g_ctrl_q, &c, 0) == pdTRUE) {
      if (c.type == T_SERVO) {
        servo.command(c.a, c.b);
      } else if (c.type == T_LED) {
        led_state = c.a <= 2 ? c.a : 0;
        led_cmd_seen = true;
      }
    }

    // ---- servo + amp rail (PWR_EN, PCB only): off until the host has spoken, see power_policy.h
    rail.tick(now, usb_bus_active(), g_host_frames.load());
    if (rail.rail_on() != rail_pin) {
      rail_pin = rail.rail_on();
      digitalWrite(PIN_PWR_EN, rail_pin ? HIGH : LOW);
    }
    const bool loads_ready = rail.loads_ready(now);

    // ---- servo: no pulses until the first SERVO command and until its supply rail has ramped.
    if (servo.active() && loads_ready) {
      if (!servo_attached) {
        servo_attached = ledcAttach(PIN_SERVO, SERVO_HZ, SERVO_RES_BITS);
        last_duty = 0xFFFFFFFF;
      }
      uint32_t duty = ServoRamp::us_to_duty(ServoRamp::cd_to_us(servo.tick()));
      if (servo_attached && duty != last_duty) {
        ledcWrite(PIN_SERVO, duty);
        last_duty = duty;
      }
    } else if (servo_attached && !loads_ready) {
      ledcDetach(PIN_SERVO);  // rail went off (USB suspend / unplug): stop pulsing, hold the line low
      pinMode(PIN_SERVO, OUTPUT);
      digitalWrite(PIN_SERVO, LOW);
      servo_attached = false;
    }

    // ---- LEDs. GPIO14 as in duet_io; the on-board RGB LED as in sketch 1, plus red for an audio error.
    const bool host_alive = g_host_seen.load() && (now - g_last_host_ms.load() < HOST_TIMEOUT_MS);
    const bool audio_failed = g_audio_err.load() != 0;
    bool led;
    if (audio_failed) {
      led = (now / 50) & 1;  // 10 Hz: audio init failed
      led_rgb(led ? 24 : 0, 0, 0);
    } else if (host_alive && led_cmd_seen) {
      led = led_state == 1 || (led_state == 2 && ((now / 250) & 1));  // 2 = 2 Hz blink
      led_rgb(0, led ? 24 : 0, 0);                                     // green: the host's LED command
    } else {
      led = (now % 1000) < 50;  // heartbeat: firmware alive, no host control
      if (host_alive) led_rgb(0, 0, 12);  // steady dim blue: host is talking
      else led_rgb(0, 0, led ? 12 : 0);   // blue blip each second: alive, no host
    }
    digitalWrite(PIN_LED, led ? HIGH : LOW);

    // ---- amplifier enable: on while accepted speaker frames are arriving, off after 2 s idle or at once
    // when the host closes the speaker gate. MAX98357A turn-on is 7-7.5 ms; the jitter buffer primes for
    // 40 ms and DMA adds 10 ms, so the first samples are never cut.
    const bool want_amp = loads_ready && g_audio_ready.load() && g_spk_on.load() && g_spk_seen.load() &&
                          (now - g_last_spk_ms.load() < AMP_IDLE_OFF_MS);
    if (want_amp != amp_on) {
      digitalWrite(PIN_AMP_SD, want_amp ? HIGH : LOW);
      amp_on = want_amp;
    }

    // ---- STATUS every 100 ms, only while the host has a stream on
    const bool streaming = g_mic_on.load() || g_spk_on.load();
    if (streaming && tick % (STATUS_PERIOD_MS / CONTROL_PERIOD_MS) == 0) {
      uint8_t p[5];
      p[0] = g_jb.level();
      put_u16(p + 1, (uint16_t)mic_overruns());
      put_u16(p + 3, (uint16_t)g_jb.underruns.load());
      post_frame(T_STATUS, seq_status++, p, 5);
    }
#if DUET_SEND_DIAG
    if (streaming && tick % 100 == 0) {
      // Parser counters are plain u32 written by the RX task; an aligned 32-bit read is atomic on this chip.
      const ParserStats& ps = g_engine.parser_stats();
      uint8_t p[16];
      put_u16(p + 0, (uint16_t)ps.bad_checksum);
      put_u16(p + 2, (uint16_t)ps.bad_header);
      put_u16(p + 4, (uint16_t)g_jb.overflows.load());
      put_u16(p + 6, (uint16_t)g_spk_seq_gaps.load());
      put_u16(p + 8, (uint16_t)g_jb.trims.load());
      put_u16(p + 10, (uint16_t)g_mic_dma_ovf);
      put_u16(p + 12, (uint16_t)(g_mic_q_drop.load() + g_mic_usb_drop.load()));
      put_u16(p + 14, (uint16_t)g_audio_err.load());
      post_frame(T_DIAG, seq_diag++, p, 16);
    }
#else
    (void)seq_diag;
#endif
  }
}

// ------------------------------------------------------------------ Arduino entry points
void setup() {
  pinMode(PIN_PWR_EN, OUTPUT);
  digitalWrite(PIN_PWR_EN, LOW);  // servo + amp rail off until the host has spoken (power_policy.h)
  pinMode(PIN_AMP_SD, OUTPUT);
  digitalWrite(PIN_AMP_SD, LOW);  // amp off until the host turns the speaker on and audio arrives
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  esp_log_level_set("*", ESP_LOG_NONE);  // keep the USB stream binary-only
  Serial.setRxBufferSize(USB_RX_BUF_BYTES);
  Serial.setTxBufferSize(USB_TX_BUF_BYTES);
  Serial.setTxTimeoutMs(0);  // never block a task on a host that is not reading
  Serial.begin();            // USB-Serial/JTAG ISR is allocated on this core (USB_CORE)
  Serial.setDebugOutput(false);

  g_mic_q = xQueueCreate(MIC_QUEUE_FRAMES, sizeof(MicItem));
  g_reply_q = xQueueCreate(REPLY_QUEUE_DEPTH, sizeof(TxFrame));
  g_ctrl_q = xQueueCreate(16, sizeof(CtrlCmd));

  g_engine.begin(engine_write, now_us, chip_info, nullptr, FW_NAME, FW_VERSION);
  g_engine.set_passthrough(on_other_frame, nullptr);

  // Order matters: the TX and audio task handles must exist before the RX task can use them.
  // The audio task only waits here; it installs nothing until the host asks.
  xTaskCreatePinnedToCore(audio_task, "audio", TASK_STACK_AUDIO, nullptr, PRIO_AUDIO, &g_audio_task, AUDIO_CORE);
  xTaskCreatePinnedToCore(usb_tx_task, "usb_tx", TASK_STACK_TX, nullptr, PRIO_USB_TX, &g_usb_tx_task, USB_CORE);
  xTaskCreatePinnedToCore(control_task, "ctrl", TASK_STACK_CTRL, nullptr, PRIO_CONTROL, nullptr, USB_CORE);
  xTaskCreatePinnedToCore(usb_rx_task, "usb_rx", TASK_STACK_RX, nullptr, PRIO_USB_RX, nullptr, USB_CORE);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));  // all work happens in the tasks above
}
