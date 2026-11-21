// Mac stand-in for the board. It is the same DuetCore the sketch compiles, with stdin as "bytes from the
// host" and stdout as "bytes to the host". host/duet_host/link.py starts it as a child process, so every
// check that will run against the real board tomorrow can run tonight against exactly the same C++.
// What it cannot show: USB behaviour, timing on the real chip, PSRAM, anything electrical.
//
// Audio: by default it behaves like sketch 2 built with DUET_AUDIO_STANDIN=1 (audio=standin): the stand-in
// microphone and speaker of DuetCore (duet_standin.h) and the sketch's own jitter buffer
// (firmware/duet_base_audio/jitter_buffer.cpp, compiled unchanged), ticked every 10 ms on an absolute schedule
// of the Mac's steady clock. Audio is off until AUDIO_CTRL, as on the board.
//   board_sim               sketch 2, stand-in audio
//   board_sim --no-audio    sketch 1: no audio; AUDIO_CTRL, SPK_AUDIO, SERVO, AUDIO_QUERY are ignored
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <chrono>

#include "DuetCore.h"
#include "jitter_buffer.h"  // from firmware/duet_base_audio (includes its config.h)

using namespace duet;
using Clock = std::chrono::steady_clock;

static_assert(STANDIN_FRAME == FRAME_SAMPLES, "DuetCore stand-in frame must equal the sketch's frame");

static uint32_t now_us() {
  static const auto t0 = Clock::now();
  return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
}

static void write_all(const uint8_t* d, size_t n, void*) {
  while (n) {
    ssize_t k = ::write(1, d, n);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) _exit(0);  // the host went away
    d += k;
    n -= (size_t)k;
  }
}

// ------------------------------------------------------------------ the simulated audio side of sketch 2
static bool g_audio = true;     // false with --no-audio
static Engine g_engine;
static JitterBuffer g_jb;
static StandinSpeaker g_spk;
static bool g_mic_on = false, g_spk_on = false, g_started = false;
static uint32_t g_mic_pos = 0;  // next stand-in sample index; 0 when the mic gate opens
static uint8_t g_mic_seq = 0;   // 0 when the mic gate opens
static uint32_t g_mic_sent = 0, g_spk_rx = 0, g_spk_refused = 0, g_spk_seq_gaps = 0;
static bool g_have_spk_seq = false;
static uint8_t g_last_spk_seq = 0;
static uint8_t g_seq_status = 0;
static uint32_t g_ticks = 0;
static Clock::time_point g_next_tick;

static size_t info(char* buf, size_t cap, void*) {
  if (!g_audio) return (size_t)snprintf(buf, cap, "chip=sim;flash=0;psram=0;sim=1");
  return (size_t)snprintf(buf, cap, "chip=sim;flash=0;psram=0;sim=1;audio=standin;audio_ready=%d;mic=%d;spk=%d",
                          g_started ? 1 : 0, g_mic_on ? 1 : 0, g_spk_on ? 1 : 0);
}

static void send_stats(uint8_t seq) {
  AudioStats s;
  memset(&s, 0, sizeof(s));
  s.mode = AUDIO_MODE_STANDIN;
  s.flags = (uint8_t)((g_mic_on ? AS_MIC_ON : 0) | (g_spk_on ? AS_SPK_ON : 0) | (g_started ? AS_RUNNING : 0) |
                      AS_SPK_SUMS);
  s.spk_level = g_jb.level();
  s.spk_state = (uint8_t)g_jb.state();
  s.mic_sent = g_mic_sent;
  s.mic_dropped = 0;  // stdout writes block instead of dropping
  s.spk_rx = g_spk_rx;
  s.spk_played = g_spk.frames.load();
  s.spk_silent = g_spk.silent.load();
  s.spk_underruns = g_jb.underruns.load();
  s.spk_dropped = g_jb.overflows.load() + g_jb.trims.load() + g_spk_refused;
  s.spk_seq_gaps = g_spk_seq_gaps;
  s.spk_crc = g_spk.crc.load();
  s.spk_peak = (uint16_t)g_spk.peak.load();
  s.dev_us = now_us();
  uint8_t p[AUDIO_STATS_LEN];
  pack_audio_stats(s, p);
  g_engine.send(T_AUDIO_STATS, seq, p, AUDIO_STATS_LEN);
}

// The same decisions as on_other_frame() in duet_base_audio.ino.
static void on_other_frame(const Frame& f, void*) {
  if (!g_audio) return;  // sketch 1 ignores every frame the engine does not answer
  switch (f.type) {
    case T_SPK_AUDIO: {
      if (!g_spk_on || !g_started) {
        ++g_spk_refused;
        break;
      }
      if (g_have_spk_seq && (uint8_t)(g_last_spk_seq + 1) != f.seq) ++g_spk_seq_gaps;
      g_have_spk_seq = true;
      g_last_spk_seq = f.seq;
      int16_t pcm[FRAME_SAMPLES];
      memcpy(pcm, f.payload, sizeof(pcm));
      if (g_jb.push(pcm)) ++g_spk_rx;
      break;
    }
    case T_AUDIO_CTRL: {
      const bool mic = (f.payload[0] & 0x01) != 0, spk = (f.payload[0] & 0x02) != 0;
      if (spk && !g_spk_on) g_have_spk_seq = false;
      if (mic && !g_mic_on) {
        g_mic_pos = 0;
        g_mic_seq = 0;
      }
      g_mic_on = mic;
      g_spk_on = spk;
      if ((mic || spk) && !g_started) {
        g_started = true;
        g_next_tick = Clock::now() + std::chrono::milliseconds(10);
      }
      break;
    }
    case T_AUDIO_QUERY: send_stats(f.seq); break;
    default: break;  // SERVO, LED: nothing to drive here
  }
}

// One 10 ms audio period: the speaker takes a frame, the microphone makes one.
static void tick() {
  static int16_t frame[FRAME_SAMPLES];
  ++g_ticks;
  if (g_jb.pop(frame)) {
    g_spk.play(frame);
  } else {
    g_spk.silence();
  }
  if (g_mic_on) {
    standin_mic_fill(g_mic_pos, frame, FRAME_SAMPLES);
    g_mic_pos += FRAME_SAMPLES;
    g_engine.send(T_MIC_AUDIO, g_mic_seq++, (const uint8_t*)frame, FRAME_SAMPLES * 2);  // little-endian Mac
    ++g_mic_sent;
  }
  if ((g_mic_on || g_spk_on) && g_ticks % (STATUS_PERIOD_MS / 10) == 0) {
    uint8_t p[5];
    p[0] = g_jb.level();
    put_u16(p + 1, 0);
    put_u16(p + 3, (uint16_t)g_jb.underruns.load());
    g_engine.send(T_STATUS, g_seq_status++, p, 5);
  }
}

int main(int argc, char** argv) {
  for (int k = 1; k < argc; ++k) {
    if (strcmp(argv[k], "--no-audio") == 0) {
      g_audio = false;
    } else {
      fprintf(stderr, "usage: board_sim [--no-audio]\n");
      return 2;
    }
  }
  g_engine.begin(write_all, now_us, info, nullptr, g_audio ? "board_sim" : "board_sim_noaudio", "0.2.0");
  g_engine.set_passthrough(on_other_frame, nullptr);
  static uint8_t buf[4096];
  struct pollfd pfd = {0, POLLIN, 0};
  for (;;) {
    int timeout_ms = 20;
    if (g_started) {
      const auto wait = std::chrono::duration_cast<std::chrono::microseconds>(g_next_tick - Clock::now()).count();
      timeout_ms = wait <= 0 ? 0 : (int)((wait + 999) / 1000);
      if (timeout_ms > 20) timeout_ms = 20;
    }
    int r = poll(&pfd, 1, timeout_ms);
    if (r < 0 && errno != EINTR) return 1;
    if (r > 0) {
      ssize_t n = ::read(0, buf, sizeof(buf));
      if (n < 0 && errno != EINTR) return 1;
      if (n == 0) return 0;  // stdin closed: the host is done
      if (n > 0) g_engine.feed(buf, (size_t)n);
    } else {
      g_engine.idle();
    }
    // Every pass, not only on a poll timeout: a steady stream of SPK_AUDIO must not starve the audio clock.
    if (g_started) {
      while (Clock::now() >= g_next_tick) {
        tick();
        g_next_tick += std::chrono::milliseconds(10);
      }
    }
  }
}
