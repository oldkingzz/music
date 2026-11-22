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
//   board_sim --live        sketch duet_live: the same Engine and LiveChain (DuetCore duet_live.h) as the board.
//                           The microphone task becomes a 16 ms steady-clock schedule: with LC_FAKE it runs the
//                           fake melody, without it a "real microphone" that is not wired (silence). The speaker
//                           task becomes a counting sink paced at 256 frames / 22050 Hz: it renders the same
//                           LiveVoice synth blocks and keeps their count, peak |sample| and CRC-32, reported in
//                           INFO as sink_blocks, sink_peak, sink_crc. Differences from the board: one thread, so no
//                           lock is needed; the sink starts and stops at once (no I2S begin time); its stop fade
//                           runs at once instead of at the speaker's pace. AUDIO_CTRL, SPK_AUDIO, SERVO, LED and
//                           AUDIO_QUERY are ignored, as in the duet_live sketch.
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
static bool g_live = false;     // --live: sketch duet_live
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

// ------------------------------------------------------------------ the simulated side of sketch duet_live
// The same constants as the top of firmware/duet_live/duet_live.ino (the team's wiring).
static const int MIC_PIN_BCLK = 11, MIC_PIN_WS = 12, MIC_PIN_SD = 13;
static const int SPK_PIN_BCLK = 4, SPK_PIN_LRC = 5, SPK_PIN_DIN = 6;
static const uint32_t MIC_RATE = PT_SAMPLE_RATE, SPK_RATE = SY_PLAY_RATE;

static LiveChain g_chain;
static LiveVoice g_voice;
static bool g_hops_running = false, g_sink_up = false;
static uint32_t g_epoch_seen = 0;
static uint64_t g_hop_n = 0, g_blk_n = 0;
static Clock::time_point g_hop_origin, g_blk_origin;
static uint32_t g_sink_crc = 0, g_sink_peak = 0;

static Clock::time_point hop_due(uint64_t n) { return g_hop_origin + std::chrono::milliseconds(LIVE_HOP_MS * n); }
static Clock::time_point blk_due(uint64_t n) {
  return g_blk_origin + std::chrono::nanoseconds(n * (uint64_t)SY_PLAY_BLOCK * 1000000000ull / SPK_RATE);
}

static size_t live_info_sim(char* buf, size_t cap, void*) {
  LiveInfoFacts f{"sim", 0, 0, 0, 0, 0, 0, -1, 0, 0, MIC_RATE, SPK_RATE,
                  {MIC_PIN_BCLK, MIC_PIN_WS, MIC_PIN_SD}, {SPK_PIN_BCLK, SPK_PIN_LRC, SPK_PIN_DIN}};
  size_t n = live_info_head(buf, cap, f);
  char tmp[96];
  snprintf(tmp, sizeof(tmp), ";chip=sim;sim=1;sink_blocks=%u;sink_peak=%u;sink_crc=%08x",
           (unsigned)g_voice.blocks.load(), (unsigned)g_sink_peak, (unsigned)g_sink_crc);
  return info_append(buf, cap, n, tmp);
}

static void live_hw(LiveHw* hw, void*) {
  hw->mic_up = g_chain.want_mic();  // the unwired "real microphone" is up as soon as it is asked for
  hw->spk_up = g_chain.singing();   // the sink starts at once
  hw->error = false;
}

static void live_frame(const Frame& f, void*) { g_chain.on_frame(f); }  // LIVE_CTRL, SYNTH_NOTE; the rest is ignored

static void sink_play(const int16_t* lr) {
  for (int i = 0; i < 2 * SY_PLAY_BLOCK; ++i) {
    const uint32_t a = (uint32_t)(lr[i] < 0 ? -(int32_t)lr[i] : lr[i]);
    if (a > g_sink_peak) g_sink_peak = a;
  }
  g_sink_crc = crc32_samples(g_sink_crc, lr, 2 * SY_PLAY_BLOCK);
}

// What the board's microphone and speaker tasks do, on the steady clock. Call every pass.
static void live_tick() {
  const auto now = Clock::now();
  // microphone: one hop per 16 ms from the moment the source (re)started
  if (g_chain.listening()) {
    if (!g_hops_running || g_chain.epoch() != g_epoch_seen) {
      g_hops_running = true;
      g_epoch_seen = g_chain.epoch();
      g_hop_origin = now;
      g_hop_n = 0;
    }
    static int32_t silence[PT_HOP];  // a real microphone that is not wired
    while (g_chain.listening() && g_chain.epoch() == g_epoch_seen && Clock::now() >= hop_due(g_hop_n + 1)) {
      if (g_chain.fake()) g_chain.hop_fake();
      else g_chain.hop_raw(silence);
      ++g_hop_n;
    }
  } else {
    g_hops_running = false;
  }
  // speaker: the counting sink
  static int16_t lr[2 * SY_PLAY_BLOCK];
  const bool want = g_chain.singing();
  if (want && !g_sink_up) {
    g_sink_up = true;
    g_voice.start();
    g_blk_origin = now;
    g_blk_n = 0;
  }
  if (g_sink_up && want) {
    while (Clock::now() >= blk_due(g_blk_n + 1)) {
      g_voice.block(lr, g_chain.gain());
      sink_play(lr);
      ++g_blk_n;
    }
  } else if (g_sink_up && !want) {
    g_voice.begin_stop();
    while (g_voice.stop_block(lr, g_chain.gain())) sink_play(lr);
    g_sink_up = false;
  } else {
    g_voice.drop_pending();
  }
}

static int live_timeout_ms() {
  auto next = Clock::now() + std::chrono::milliseconds(20);
  if (g_chain.listening() && g_hops_running && hop_due(g_hop_n + 1) < next) next = hop_due(g_hop_n + 1);
  if (g_sink_up && blk_due(g_blk_n + 1) < next) next = blk_due(g_blk_n + 1);
  if (g_chain.listening() && !g_hops_running) return 0;
  const auto wait = std::chrono::duration_cast<std::chrono::microseconds>(next - Clock::now()).count();
  return wait <= 0 ? 0 : (int)((wait + 999) / 1000);
}

static int live_main() {
  g_engine.begin(write_all, now_us, live_info_sim, nullptr, "duet_live", "0.3.0");
  g_engine.set_passthrough(live_frame, nullptr);
  g_chain.begin(&g_engine, &g_voice, live_hw, nullptr);
  static uint8_t buf[4096];
  struct pollfd pfd = {0, POLLIN, 0};
  for (;;) {
    int r = poll(&pfd, 1, live_timeout_ms());
    if (r < 0 && errno != EINTR) return 1;
    if (r > 0) {
      ssize_t n = ::read(0, buf, sizeof(buf));
      if (n < 0 && errno != EINTR) return 1;
      if (n == 0) return 0;  // stdin closed: the host is done
      if (n > 0) g_engine.feed(buf, (size_t)n);
    } else {
      g_engine.idle();
    }
    live_tick();  // every pass, so a steady stream of host frames cannot starve the audio clock
  }
}

int main(int argc, char** argv) {
  for (int k = 1; k < argc; ++k) {
    if (strcmp(argv[k], "--no-audio") == 0) {
      g_audio = false;
    } else if (strcmp(argv[k], "--live") == 0) {
      g_live = true;
    } else {
      fprintf(stderr, "usage: board_sim [--no-audio | --live]\n");
      return 2;
    }
  }
  if (g_live) return live_main();
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
