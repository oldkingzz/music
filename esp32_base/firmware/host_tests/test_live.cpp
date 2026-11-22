// Host unit tests of the live chain of sketch duet_live (DuetCore: duet_pitch, duet_synth, duet_fake16,
// duet_live, the new frames). Built with sanitizers by build_and_test.sh.
//   test_live            the tests
//   test_live --dump     also print the measured numbers (event lists, delays, accuracy, CPU time)
//
// The pinned event lists below were taken from what the lifted tracker does (test_live --dump), then pinned.
// host/checks/check_live.py and host/tests/test_live.py pin the same list.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "DuetCore.h"

using namespace duet;

static int g_checks = 0, g_fail = 0;
static bool g_dump = false;
#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_fail;                                                                  \
      printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                     \
    }                                                                            \
  } while (0)

struct Got {
  uint8_t type, seq;
  std::vector<uint8_t> payload;
};
static void collect(const Frame& f, void* ctx) {
  ((std::vector<Got>*)ctx)->push_back(Got{f.type, f.seq, std::vector<uint8_t>(f.payload, f.payload + f.len)});
}
static std::vector<uint8_t> frame(uint8_t type, uint8_t seq, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out(HEADER_LEN + payload.size() + 1);
  size_t n = encode(type, seq, payload.data(), (uint16_t)payload.size(), out.data());
  out.resize(n);
  return out;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ------------------------------------------------------------------ 1. constants equal the team's sketch
static void test_constants() {
  // quantization_code_4 lines 62-90 (pitch) and 99-110 (playback)
  CHECK(PT_SAMPLE_RATE == 16000);
  CHECK(PT_FRAME == 768);
  CHECK(PT_HOP == 256);
  CHECK(PT_F_MIN == 75.0f);
  CHECK(PT_F_MAX == 1100.0f);
  CHECK(PT_TAU_MIN == (int)(16000u / 1100.0f) && PT_TAU_MIN == 14);
  CHECK(PT_TAU_MAX == (int)(16000u / 75.0f) && PT_TAU_MAX == 213);
  CHECK(PT_W == 768 - 213);
  CHECK(PT_YIN_THRESHOLD == 0.15f);
  CHECK(PT_MIN_CONFIDENCE == 0.60f);
  CHECK(PT_RMS_GATE == 0.004f);
  CHECK(PT_HYSTERESIS == 0.65f);
  CHECK(PT_SETTLE_FRAMES == 5);
  CHECK(PT_SETTLE_SPREAD == 0.6f);
  CHECK(PT_RELEASE_FRAMES == 6);
  CHECK(SY_PLAY_RATE == 22050);
  CHECK(SY_PLAY_BLOCK == 256);
  CHECK(SY_PLAY_VOLUME == 0.35f);
  CHECK(SY_NUM_VOICES == 4);
  CHECK(SY_WAVE_BITS == 11 && SY_WAVE_N == 2048);
  CHECK(SY_ATTACK_SEC == 0.005f);
  CHECK(SY_DECAY_TAU_SEC == 1.2f);
  CHECK(SY_DECAY_FLOOR == 0.25f);
  CHECK(SY_RELEASE_TAU_SEC == 0.06f);
  CHECK(SY_FAST_RELEASE_SEC == 0.008f);
  CHECK(SY_STOP_FADE_BLOCKS == 30);
  // their 24-bit conversion: (raw >> 8) / 8388608.0f; an int16 x fed as x << 16 gives x / 32768 exactly
  CHECK(PitchTracker::raw_to_i16(12345 * 65536) == 12345);
  CHECK(PitchTracker::raw_to_i16(-32768 * 65536) == -32768);
  CHECK(PitchTracker::raw_to_i16((int32_t)0x7FFFFF00) == 32767);  // saturates (rounding would give 32768)
  CHECK(PitchTracker::raw_to_i16(-1) == 0);
}

// ------------------------------------------------------------------ 2. the 16 kHz fake microphone
// The same golden values are asserted in host/tests/test_live.py (Python mirror).
static const uint32_t FAKE16_LOOP_CRC = 0xa35c56edu;  // CRC-32 of samples 0..127999 (int16 LE bytes)
static void test_fake16() {
  std::vector<int16_t> x(FAKE16_LOOP);
  fake16_fill(0, x.data(), x.size());
  uint32_t crc = crc32_samples(0, x.data(), x.size());
  int64_t sum = 0;
  int peak = 0;
  for (int16_t v : x) {
    sum += v;
    peak = std::max(peak, std::abs((int)v));
  }
  if (g_dump) {
    printf("fake16: first");
    for (int i = 0; i < 8; ++i) printf(" %d", x[i]);
    printf("; known");
    for (uint32_t n : {100u, 1000u, 5000u, 12345u, 127999u}) printf(" %u:%d", n, x[n]);
    printf("; sum %lld peak %d crc 0x%08x\n", (long long)sum, peak, (unsigned)crc);
  }
  CHECK(crc == FAKE16_LOOP_CRC);
  const int16_t first[8] = {0, 36, 136, 283, 463, 660, 858, 1042};
  CHECK(memcmp(x.data(), first, sizeof(first)) == 0);
  CHECK(x[100] == -3973 && x[1000] == 3859 && x[5000] == -5844 && x[12345] == 7408 && x[127999] == 0);
  CHECK(sum == 37410 && peak == 11999);
  // a pure function of n: the second loop repeats the first
  bool same = true;
  for (uint32_t n = 0; n < FAKE16_LOOP; n += 7) same &= fake16_sample(n + FAKE16_LOOP) == x[n];
  CHECK(same);
  // the gap is silent, the notes are not
  uint32_t start = 0;
  bool gaps_silent = true, notes_loud = true;
  for (size_t k = 0; k < FAKE16_NOTES; ++k) {
    const uint32_t slot = FAKE16_MELODY[k].slot;
    for (uint32_t t = slot - FAKE16_GAP; t < slot; ++t) gaps_silent &= x[start + t] == 0;
    int pk = 0;
    for (uint32_t t = 0; t < slot - FAKE16_GAP; ++t) pk = std::max(pk, std::abs((int)x[start + t]));
    notes_loud &= pk > 10000;
    start += slot;
  }
  CHECK(start == FAKE16_LOOP);
  CHECK(gaps_silent && notes_loud);
  CHECK(FAKE16_GAP == 2560 && FAKE16_GAP_50MS == 800);
}

// ------------------------------------------------------------------ 3. YIN on pure tones
static void test_yin() {
  PitchTracker t;
  std::vector<float> fr(PT_FRAME);
  double worst = 0, worst_conf = 1;
  int worst_m = 0;
  for (int m = 45; m <= 84; ++m) {  // A2 (110 Hz) .. C6 (1047 Hz), the singing range
    const double f = 440.0 * pow(2.0, (m - 69) / 12.0);
    for (double phase0 : {0.0, 1.3}) {
      for (int i = 0; i < PT_FRAME; ++i) fr[i] = (float)(0.3 * sin(2 * M_PI * f * i / PT_SAMPLE_RATE + phase0));
      float conf = 0;
      const float got = t.yin(fr.data(), conf);
      const double cents = got > 0 ? 1200.0 * log2(got / f) : 9999;
      if (fabs(cents) > worst) {
        worst = fabs(cents);
        worst_m = m;
      }
      worst_conf = std::min(worst_conf, (double)conf);
    }
  }
  if (g_dump) printf("yin: pure tones MIDI 45..84: worst error %.2f cents (MIDI %d), lowest confidence %.3f\n", worst,
                     worst_m, worst_conf);
  CHECK(worst < 10.0);
  CHECK(worst_conf > 0.95);
  // silence: no pitch
  for (int i = 0; i < PT_FRAME; ++i) fr[i] = 0;
  float conf = 1;
  t.yin(fr.data(), conf);
  CHECK(conf == 0.0f);
}

// ------------------------------------------------------------------ 4. the tracker on the fake melody
struct Ev {
  uint8_t kind;  // PT_NOTE_ON / PT_NOTE_OFF
  uint8_t midi;
  uint32_t hop;  // 1-based: number of hops processed including the one that gave the event
};
static std::vector<Ev> run_tracker(uint32_t gap, uint32_t hops) {
  PitchTracker t;
  std::vector<Ev> out;
  int16_t h[PT_HOP];
  PitchEvent ev[PitchTracker::MAX_EVENTS];
  for (uint32_t k = 0; k < hops; ++k) {
    fake16_fill(k * PT_HOP, h, PT_HOP, gap);
    const int n = t.hop_i16(h, ev);
    for (int i = 0; i < n; ++i) out.push_back(Ev{ev[i].kind, ev[i].midi, k + 1});
  }
  return out;
}
static void dump_events(const char* name, const std::vector<Ev>& v) {
  printf("%s (%zu events): {", name, v.size());
  for (size_t i = 0; i < v.size(); ++i) printf("%s{%u,%u,%u}", i ? "," : "", v[i].kind, v[i].midi, v[i].hop);
  printf("}\n");
}

// One 8 s loop plus 30 hops (the first note of the next loop, on and off): {kind, midi, hop}. 1 = on, 2 = off.
static const Ev PIN_160[] = {
    {1,60,6}, {2,60,29}, {1,60,37}, {2,60,61}, {1,67,68}, {2,67,92}, {1,67,99}, {2,67,123},
    {1,69,131}, {2,69,155}, {1,69,162}, {2,69,186}, {1,67,193}, {2,67,248}, {1,65,256}, {2,65,279},
    {1,65,287}, {2,65,311}, {1,64,318}, {2,64,342}, {1,64,349}, {2,64,373}, {1,62,381}, {2,62,404},
    {1,62,412}, {2,62,436}, {1,60,443}, {2,60,498}, {1,60,506}, {2,60,529},
};
// Gap 50 ms (the old stand-in gap): repeated notes merge. One loop plus 30 hops.
static const Ev PIN_50[] = {
    {1,60,6}, {2,60,68}, {1,67,68}, {2,67,131}, {1,69,131}, {2,69,193}, {1,67,193}, {2,67,256},
    {1,65,256}, {2,65,318}, {1,64,318}, {2,64,381}, {1,62,381}, {2,62,443}, {1,60,443},
};

static const uint32_t TRACK_HOPS = FAKE16_LOOP / PT_HOP + 30;  // 530: one loop + the first note of the next

static void test_tracker_melody() {
  const std::vector<Ev> got = run_tracker(FAKE16_GAP, TRACK_HOPS);
  if (g_dump) dump_events("tracker gap 160 ms", got);
  const size_t n_pin = sizeof(PIN_160) / sizeof(PIN_160[0]);
  bool same = got.size() == n_pin;
  for (size_t i = 0; same && i < n_pin; ++i)
    same = got[i].kind == PIN_160[i].kind && got[i].midi == PIN_160[i].midi && got[i].hop == PIN_160[i].hop;
  CHECK(same);
  // the 14 melody notes come out separately, in order, each on then off
  std::vector<int> ons;
  for (const Ev& e : got)
    if (e.kind == PT_NOTE_ON) ons.push_back(e.midi);
  const int melody[] = {60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60, 60};
  CHECK(ons == std::vector<int>(melody, melody + 15));
  // delays: true note start -> HEARD_ON hop end, true sound end -> HEARD_OFF hop end
  std::vector<double> d_on, d_off;
  uint32_t start = 0;
  size_t k_on = 0, k_off = 0;
  for (size_t k = 0; k < FAKE16_NOTES; ++k) {
    const uint32_t slot = FAKE16_MELODY[k].slot;
    const double on_ms = start / 16.0, off_ms = (start + slot - FAKE16_GAP) / 16.0;
    while (k_on < got.size() && got[k_on].kind != PT_NOTE_ON) ++k_on;
    while (k_off < got.size() && got[k_off].kind != PT_NOTE_OFF) ++k_off;
    if (k_on < got.size()) d_on.push_back(got[k_on++].hop * 16.0 - on_ms);
    if (k_off < got.size()) d_off.push_back(got[k_off++].hop * 16.0 - off_ms);
    start += slot;
  }
  CHECK(d_on.size() == 14 && d_off.size() == 14);
  const double on_min = *std::min_element(d_on.begin(), d_on.end()), on_max = *std::max_element(d_on.begin(), d_on.end());
  const double off_min = *std::min_element(d_off.begin(), d_off.end()),
               off_max = *std::max_element(d_off.begin(), d_off.end());
  if (g_dump)
    printf("tracker delays: note start -> HEARD_ON min %.0f / median %.0f / max %.0f ms; sound end -> HEARD_OFF "
           "min %.0f / median %.0f / max %.0f ms\n",
           on_min, median(d_on), on_max, off_min, median(d_off), off_max);
  CHECK(on_min >= 0 && on_max <= 160 && off_min >= 0 && off_max <= 200);

  // the 50 ms gap: repeated notes merge (a limit of the tracker, written down here)
  const std::vector<Ev> g50 = run_tracker(FAKE16_GAP_50MS, TRACK_HOPS);
  if (g_dump) dump_events("tracker gap 50 ms", g50);
  const size_t n50 = sizeof(PIN_50) / sizeof(PIN_50[0]);
  same = g50.size() == n50;
  for (size_t i = 0; same && i < n50; ++i)
    same = g50[i].kind == PIN_50[i].kind && g50[i].midi == PIN_50[i].midi && g50[i].hop == PIN_50[i].hop;
  CHECK(same);
  std::vector<int> ons50;
  for (const Ev& e : g50)
    if (e.kind == PT_NOTE_ON) ons50.push_back(e.midi);
  const int merged[] = {60, 67, 69, 67, 65, 64, 62, 60};
  CHECK(ons50 == std::vector<int>(merged, merged + 8));

  // The real-mic path (32-bit I2S words) gives the same events as the int16 path for the same samples.
  PitchTracker a, b;
  int16_t h[PT_HOP];
  int32_t raw[PT_HOP];
  PitchEvent ea[2], eb[2];
  bool equal = true;
  for (uint32_t k = 0; k < 300; ++k) {
    fake16_fill(k * PT_HOP, h, PT_HOP);
    for (int i = 0; i < PT_HOP; ++i) raw[i] = (int32_t)h[i] * 65536 + 0x55;  // low 8 bits are ignored by >> 8
    const int na = a.hop_i16(h, ea), nb = b.hop_raw(raw, eb);
    equal &= na == nb && a.active() == b.active() && a.freq() == b.freq() && a.conf() == b.conf();
    for (int i = 0; equal && i < na; ++i) equal &= ea[i].kind == eb[i].kind && ea[i].midi == eb[i].midi && ea[i].vel == eb[i].vel;
  }
  CHECK(equal);
  // silence: nothing
  PitchTracker s;
  int32_t zero[PT_HOP] = {0};
  int n = 0;
  for (int k = 0; k < 200; ++k) n += s.hop_raw(zero, ea);
  CHECK(n == 0 && s.active() == -1 && s.freq() == 0.0f && s.conf() == 0.0f);

  // CPU time per hop on this Mac (sanitizer build: slower than -O2 alone)
  PitchTracker c;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t k = 0; k < 2000; ++k) {
    fake16_fill((k % 500) * PT_HOP, h, PT_HOP);
    c.hop_i16(h, ea);
  }
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 2000;
  if (g_dump) printf("tracker CPU per hop on this Mac (ASan/UBSan build): %.1f us\n", us);
}

// ------------------------------------------------------------------ 5. the synth
// Frequency of a periodic signal by the normalised difference function (double precision), around f0.
static double measure_freq(const std::vector<float>& x, double rate, double f0) {
  const int p0 = (int)(rate / f0);
  const int lo = std::max(2, (int)(p0 * 0.8)), hi = (int)(p0 * 1.25) + 2;
  const int W = (int)x.size() - hi - 1;
  std::vector<double> d(hi + 2, 0);
  for (int tau = lo - 1; tau <= hi + 1; ++tau) {
    double s = 0;
    for (int j = 0; j < W; ++j) {
      const double e = x[j] - x[j + tau];
      s += e * e;
    }
    d[tau] = s;
  }
  int best = lo;
  for (int tau = lo; tau <= hi; ++tau)
    if (d[tau] < d[best]) best = tau;
  const double s0 = d[best - 1], s1 = d[best], s2 = d[best + 1];
  const double den = 2 * (2 * s1 - s2 - s0);
  const double tau = best + (fabs(den) > 0 ? (s2 - s0) / den : 0);
  return rate / tau;
}

static void test_synth() {
  // pitch of a rendered note
  double worst = 0;
  for (int m : {48, 57, 60, 64, 67, 69, 72, 76, 81}) {
    Synth s;
    s.note_on((uint8_t)m, 100);
    std::vector<int16_t> mono(SY_PLAY_RATE / 2);
    s.render(mono.data(), mono.size(), false, SY_PLAY_VOLUME);
    std::vector<float> x(mono.size() - 2000);
    for (size_t i = 0; i < x.size(); ++i) x[i] = mono[i + 2000];
    const double f = 440.0 * pow(2.0, (m - 69) / 12.0);
    const double got = measure_freq(x, SY_PLAY_RATE, f);
    worst = std::max(worst, fabs(1200.0 * log2(got / f)));
  }
  if (g_dump) printf("synth: pitch of rendered notes MIDI 48..81: worst error %.3f cents\n", worst);
  CHECK(worst < 3.0);

  // peak within the volume cap, sound, then silence after the release; no NaN / inf
  {
    Synth s;
    s.note_on(69, 127);
    std::vector<int16_t> lr(2 * SY_PLAY_BLOCK);
    int peak = 0;
    bool stereo_same = true;
    for (int b = 0; b < 40; ++b) {
      s.render(lr.data(), SY_PLAY_BLOCK, true, SY_PLAY_VOLUME);
      for (int i = 0; i < SY_PLAY_BLOCK; ++i) {
        peak = std::max(peak, std::abs((int)lr[2 * i]));
        stereo_same &= lr[2 * i] == lr[2 * i + 1];
      }
    }
    const int cap = (int)(SY_PLAY_VOLUME * 32767.0f);
    if (g_dump) printf("synth: one note at full volume, peak %d (cap %d)\n", peak, cap);
    CHECK(peak > cap / 2 && peak <= cap);
    CHECK(stereo_same);
    s.note_off(69);
    int blocks = 0;
    while (!s.idle() && blocks < 200) {
      s.render(lr.data(), SY_PLAY_BLOCK, true, SY_PLAY_VOLUME);
      ++blocks;
    }
    CHECK(s.idle() && blocks < 60);  // release tau 60 ms: about 0.46 s = 40 blocks
    s.render(lr.data(), SY_PLAY_BLOCK, true, SY_PLAY_VOLUME);
    bool silent = true;
    for (int16_t v : lr) silent &= v == 0;
    CHECK(silent);
    if (g_dump) printf("synth: idle %d blocks (%.0f ms) after note off\n", blocks, blocks * 256000.0 / SY_PLAY_RATE);
  }
  // render_sample never gives NaN / inf, voice stealing (more notes than voices) does not crash
  {
    Synth s;
    bool finite = true;
    for (int k = 0; k < 40; ++k) {
      s.note_on((uint8_t)(40 + k), (uint8_t)(k * 3));
      for (int i = 0; i < 300; ++i) finite &= std::isfinite(s.render_sample());
      if (k % 3 == 0) s.note_off((uint8_t)(40 + k));
    }
    CHECK(finite);
    s.note_on(0, 127);
    s.note_on(127, 127);
    s.note_off(5);  // not sounding: nothing happens
    std::vector<int16_t> lr(2 * SY_PLAY_BLOCK);
    for (int b = 0; b < 20; ++b) s.render(lr.data(), SY_PLAY_BLOCK, true, 1.0f);  // above the cap: clipped, no wrap
    CHECK(!s.idle());
    s.release_all_fast();
    int blocks = 0;
    while (!s.idle() && blocks < SY_STOP_FADE_BLOCKS) {
      s.render(lr.data(), SY_PLAY_BLOCK, true, SY_PLAY_VOLUME);
      ++blocks;
    }
    CHECK(s.idle() && blocks <= 8);  // fast release 8 ms: about 61 ms
  }
  // volume 0 is silent
  {
    Synth s;
    s.note_on(60, 127);
    std::vector<int16_t> lr(2 * SY_PLAY_BLOCK);
    bool silent = true;
    for (int b = 0; b < 10; ++b) {
      s.render(lr.data(), SY_PLAY_BLOCK, true, synth_gain(0));
      for (int16_t v : lr) silent &= v == 0;
    }
    CHECK(silent);
    CHECK(synth_gain(100) == SY_PLAY_VOLUME && synth_gain(250) == SY_PLAY_VOLUME && synth_gain(50) == 0.35f * 50 / 100);
  }
  // a different sample rate gives the same pitch
  {
    Synth s(48000);
    s.note_on(69, 100);
    std::vector<int16_t> mono(24000);
    s.render(mono.data(), mono.size(), false, SY_PLAY_VOLUME);
    std::vector<float> x(mono.begin() + 4000, mono.end());
    CHECK(fabs(1200.0 * log2(measure_freq(x, 48000, 440.0) / 440.0)) < 3.0);
  }
  // the command ring
  {
    SynthQueue<4> q;
    SynthCmd c;
    CHECK(!q.pop(&c));
    for (uint8_t k = 0; k < 4; ++k) CHECK(q.push(SynthCmd{SC_NOTE_ON, k, 1}));
    CHECK(!q.push(SynthCmd{SC_NOTE_ON, 9, 1}) && q.dropped() == 1);
    for (uint8_t k = 0; k < 4; ++k) CHECK(q.pop(&c) && c.midi == k);
    CHECK(!q.pop(&c));
    Synth s;
    synth_apply(s, SynthCmd{SC_NOTE_ON, 60, 90});
    CHECK(!s.idle());
    synth_apply(s, SynthCmd{SC_ALL_OFF, 0, 0});
    std::vector<int16_t> lr(2 * SY_PLAY_BLOCK);
    for (int b = 0; b < 60; ++b) s.render(lr.data(), SY_PLAY_BLOCK, true, SY_PLAY_VOLUME);
    CHECK(s.idle());
  }
}

// ------------------------------------------------------------------ 6. protocol helpers
static void test_live_protocol() {
  const struct {
    uint8_t t;
    uint16_t len;
  } types[] = {{T_LIVE_CTRL, 4}, {T_SYNTH_NOTE, 2}, {T_LIVE_NOTE, 8}, {T_LIVE_STATE, 24}, {T_MIC_HOP, 512}};
  CHECK(T_LIVE_CTRL == 0x18 && T_SYNTH_NOTE == 0x19 && T_LIVE_NOTE == 0x28 && T_LIVE_STATE == 0x29 &&
        T_MIC_HOP == 0x2A);
  for (const auto& ty : types) {
    uint16_t lo = 0, hi = 0;
    CHECK(len_range(ty.t, &lo, &hi) && lo == ty.len && hi == ty.len);
    std::vector<uint8_t> payload(ty.len);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)(i * 13 + ty.t);
    std::vector<uint8_t> fr = frame(ty.t, 200, payload);
    Parser p;
    std::vector<Got> got;
    for (uint8_t b : fr) p.feed(&b, 1, collect, &got);
    CHECK(got.size() == 1 && got[0].type == ty.t && got[0].seq == 200 && got[0].payload == payload);
    for (uint16_t bad : {(uint16_t)(ty.len - 1), (uint16_t)(ty.len + 1)}) {
      std::vector<uint8_t> fb = frame(ty.t, 1, std::vector<uint8_t>(bad, 0));
      Parser q;
      std::vector<Got> g2;
      q.feed(fb.data(), fb.size(), collect, &g2);
      CHECK(g2.empty() && q.stats().bad_header >= 1);
    }
  }
  // the old types are untouched
  uint16_t lo, hi;
  CHECK(len_range(T_MIC_AUDIO, &lo, &hi) && lo == 480 && hi == 480);
  CHECK(len_range(T_AUDIO_STATS, &lo, &hi) && lo == 48 && hi == 48);
  CHECK(!len_range(0x1A, nullptr, nullptr) && !len_range(0x2B, nullptr, nullptr));

  uint8_t b[LIVE_STATE_LEN];
  LiveCtrl c{LC_LISTEN | LC_FAKE, 77, 0}, c2;
  pack_live_ctrl(c, b);
  CHECK(b[0] == 0x05 && b[1] == 77 && b[2] == 0 && b[3] == 0);
  CHECK(parse_live_ctrl(b, 4, &c2) && c2.flags == 5 && c2.volume == 77 && c2.bpm == 0);
  CHECK(!parse_live_ctrl(b, 3, &c2));
  LiveNote n{LN_SUNG_ON, 57, 99, LN_FAKE, 0x01020304u}, n2;
  pack_live_note(n, b);
  const uint8_t want_n[8] = {3, 57, 99, 1, 4, 3, 2, 1};
  CHECK(memcmp(b, want_n, 8) == 0);
  CHECK(parse_live_note(b, 8, &n2) && n2.kind == 3 && n2.midi == 57 && n2.vel == 99 && n2.flags == 1 &&
        n2.t_ms == 0x01020304u);
  LiveState s{LS_LISTEN | LS_REPLY, 100, -1, 57, 261.5f, 97, -14, -50, 123456, 14, 13, 0xA0B0C0D0u}, s2;
  pack_live_state(s, b);
  CHECK(b[0] == 0x81 && b[2] == 0xFF && b[3] == 57 && b[9] == (uint8_t)-14 && b[10] == (uint8_t)-50 && b[11] == 0);
  CHECK(get_u32(b + 12) == 123456 && get_u16(b + 16) == 14 && get_u16(b + 18) == 13 && get_u32(b + 20) == 0xA0B0C0D0u);
  CHECK(parse_live_state(b, 24, &s2) && s2.flags == s.flags && s2.heard == -1 && s2.sung == 57 && s2.freq == 261.5f &&
        s2.conf == 97 && s2.level == -14 && s2.cents == -50 && s2.hops == 123456 && s2.heard_n == 14 &&
        s2.sung_n == 13 && s2.spk_blocks == 0xA0B0C0D0u);
}

// ------------------------------------------------------------------ 7. the chain end to end
struct Rig {
  Engine e;
  LiveVoice v;
  LiveChain ch;
  std::vector<uint8_t> tx;
  Parser host;
  std::vector<Got> got;
  uint32_t clock = 0;
  static uint32_t s_clock;
  static uint32_t micros() { return s_clock += 3; }
  static void sink(const uint8_t* d, size_t n, void* ctx) {
    Rig* r = (Rig*)ctx;
    r->host.feed(d, n, collect, &r->got);
  }
  static void pass(const Frame& f, void* ctx) { ((Rig*)ctx)->ch.on_frame(f); }
  static void hw(LiveHw* h, void* ctx) {
    Rig* r = (Rig*)ctx;
    h->mic_up = r->ch.want_mic();
    h->spk_up = r->ch.singing();
    h->error = false;
  }
  Rig() {
    e.begin(sink, micros, nullptr, this, "duet_live", "0.3.0");
    e.set_passthrough(pass, this);
    ch.begin(&e, &v, hw, this);
  }
  void send(uint8_t type, uint8_t seq, const std::vector<uint8_t>& p) {
    std::vector<uint8_t> f = frame(type, seq, p);
    e.feed(f.data(), f.size());
  }
  void ctrl(uint8_t flags, uint8_t vol = 100, uint8_t bpm = 0, uint8_t seq = 0) { send(T_LIVE_CTRL, seq, {flags, vol, bpm, 0}); }
  void config(uint8_t plugin, uint8_t tonic) { send(T_CONFIG, 0, {plugin, tonic, 4, 0, 0, 0, 0, 0}); }
  std::vector<Got> take(uint8_t type) {
    std::vector<Got> out;
    for (const Got& g : got)
      if (g.type == type) out.push_back(g);
    return out;
  }
};
uint32_t Rig::s_clock = 0;

static std::vector<LiveNote> notes_of(const std::vector<Got>& v, std::vector<uint8_t>* seqs = nullptr) {
  std::vector<LiveNote> out;
  for (const Got& g : v) {
    if (g.type != T_LIVE_NOTE) continue;
    LiveNote n;
    if (parse_live_note(g.payload.data(), (uint16_t)g.payload.size(), &n)) out.push_back(n);
    if (seqs) seqs->push_back(g.seq);
  }
  return out;
}
static int third_below_c(int midi) { return harmonizer(H_THIRD_BELOW)->step(0, 0, midi, -1); }

// Expected LIVE_NOTE list of one 8 s loop + 30 hops with third_below in C: {kind, midi, t_ms}
// (kind 1 HEARD_ON, 2 HEARD_OFF, 3 SUNG_ON, 4 SUNG_OFF). Also in host/checks/check_live.py (EXPECTED).
struct PinNote {
  uint8_t kind, midi;
  uint32_t t_ms;
};
static const PinNote PIN_CHAIN[] = {
    {1,60,96}, {3,57,96}, {2,60,464}, {4,57,464}, {1,60,592}, {3,57,592},
    {2,60,976}, {4,57,976}, {1,67,1088}, {3,64,1088}, {2,67,1472}, {4,64,1472},
    {1,67,1584}, {3,64,1584}, {2,67,1968}, {4,64,1968}, {1,69,2096}, {3,65,2096},
    {2,69,2480}, {4,65,2480}, {1,69,2592}, {3,65,2592}, {2,69,2976}, {4,65,2976},
    {1,67,3088}, {3,64,3088}, {2,67,3968}, {4,64,3968}, {1,65,4096}, {3,62,4096},
    {2,65,4464}, {4,62,4464}, {1,65,4592}, {3,62,4592}, {2,65,4976}, {4,62,4976},
    {1,64,5088}, {3,60,5088}, {2,64,5472}, {4,60,5472}, {1,64,5584}, {3,60,5584},
    {2,64,5968}, {4,60,5968}, {1,62,6096}, {3,59,6096}, {2,62,6464}, {4,59,6464},
    {1,62,6592}, {3,59,6592}, {2,62,6976}, {4,59,6976}, {1,60,7088}, {3,57,7088},
    {2,60,7968}, {4,57,7968}, {1,60,8096}, {3,57,8096}, {2,60,8464}, {4,57,8464},
};

static void test_chain() {
  {
    Rig r;
    r.config(H_THIRD_BELOW, 0);
    r.got.clear();
    r.ctrl(LC_LISTEN | LC_FAKE, 100, 0, 77);
    // the reply: same seq, LS_REPLY, gates in force
    std::vector<Got> st = r.take(T_LIVE_STATE);
    CHECK(st.size() == 1 && st[0].seq == 77);
    LiveState s;
    CHECK(!st.empty() && parse_live_state(st[0].payload.data(), 24, &s) && s.flags == (LS_LISTEN | LS_FAKE | LS_REPLY) &&
          s.heard == -1 && s.sung == -1 && s.hops == 0 && s.volume == 100);
    r.got.clear();
    for (uint32_t k = 0; k < TRACK_HOPS; ++k) r.ch.hop_fake();
    std::vector<uint8_t> seqs;
    std::vector<LiveNote> ns = notes_of(r.got, &seqs);
    if (g_dump) {
      printf("chain third_below in C (%zu events): {", ns.size());
      for (size_t i = 0; i < ns.size(); ++i) printf("%s{%u,%u,%u}", i ? "," : "", ns[i].kind, ns[i].midi, (unsigned)ns[i].t_ms);
      printf("}\n");
    }
    const size_t n_pin = sizeof(PIN_CHAIN) / sizeof(PIN_CHAIN[0]);
    bool same = ns.size() == n_pin;
    for (size_t i = 0; same && i < n_pin; ++i)
      same = ns[i].kind == PIN_CHAIN[i].kind && ns[i].midi == PIN_CHAIN[i].midi && ns[i].t_ms == PIN_CHAIN[i].t_ms;
    CHECK(same);
    // seq counts every event from 0; flags say fake; each SUNG_ON is third_below of the HEARD_ON before it
    bool seq_ok = true;
    for (size_t i = 0; i < seqs.size(); ++i) seq_ok &= seqs[i] == (uint8_t)i;
    CHECK(seq_ok);
    std::vector<int> heard, sung;
    bool pairs_ok = true, flags_ok = true;
    int last_heard = -1, last_vel = -1;
    for (const LiveNote& n : ns) {
      flags_ok &= n.flags == LN_FAKE;
      if (n.kind == LN_HEARD_ON) {
        heard.push_back(n.midi);
        last_heard = n.midi;
        last_vel = n.vel;
      }
      if (n.kind == LN_SUNG_ON) {
        sung.push_back(n.midi);
        pairs_ok &= n.midi == third_below_c(last_heard) && n.vel == last_vel;
      }
      if (n.kind == LN_HEARD_OFF || n.kind == LN_SUNG_OFF) flags_ok &= n.vel == 0;
    }
    const int melody[] = {60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60, 60};
    CHECK(heard == std::vector<int>(melody, melody + 15));
    CHECK(sung.size() == heard.size() && pairs_ok && flags_ok);
    if (g_dump) {
      printf("chain: heard");
      for (int m : heard) printf(" %d", m);
      printf("; sung");
      for (int m : sung) printf(" %d", m);
      printf("\n");
    }
    // LIVE_STATE every 6 hops, own seq from 0, hops field 6, 12, ...
    std::vector<Got> states = r.take(T_LIVE_STATE);
    CHECK(states.size() == TRACK_HOPS / 6);
    bool st_ok = true;
    for (size_t i = 0; i < states.size(); ++i) {
      LiveState x;
      st_ok &= parse_live_state(states[i].payload.data(), 24, &x) && states[i].seq == (uint8_t)i &&
               x.hops == 6 * (i + 1) && !(x.flags & LS_REPLY) && x.level <= 0 && x.level >= -100 && x.conf <= 100;
    }
    CHECK(st_ok);
    // SING is off: SUNG events are reported, but nothing goes to the synth
    SynthCmd c;
    CHECK(!r.v.q.pop(&c));
    CHECK(r.take(T_MIC_HOP).empty());
    // a live LIVE_STATE in the middle of a note: heard and sung set, freq near the note
    r.got.clear();
    for (int k = 0; k < 20; ++k) r.ch.hop_fake();  // hop 560: inside note 2 (60) of loop 2
    r.ctrl(LC_LISTEN | LC_FAKE, 100, 0, 5);
    st = r.take(T_LIVE_STATE);
    CHECK(!st.empty());
    LiveState mid;
    CHECK(!st.empty() && parse_live_state(st.back().payload.data(), 24, &mid) && (mid.flags & LS_REPLY) &&
          st.back().seq == 5 && mid.hops == TRACK_HOPS + 20);
    CHECK(mid.heard == 60 && mid.sung == third_below_c(60) && fabsf(mid.freq - 261.63f) < 3.0f && mid.conf > 90 &&
          mid.level > -30 && mid.level < 0 && mid.cents >= -50 && mid.cents <= 50);
    // LISTEN off: the sounding notes end first (HEARD_OFF, SUNG_OFF), then the reply; counters stay
    r.got.clear();
    r.ctrl(0, 100, 0, 6);
    CHECK(r.got.size() == 3 && r.got[0].type == T_LIVE_NOTE && r.got[1].type == T_LIVE_NOTE &&
          r.got[2].type == T_LIVE_STATE && r.got[2].seq == 6);
    std::vector<LiveNote> off = notes_of(r.got);
    CHECK(off.size() == 2 && off[0].kind == LN_HEARD_OFF && off[0].midi == 60 && off[1].kind == LN_SUNG_OFF &&
          off[1].midi == third_below_c(60) && off[0].t_ms == (TRACK_HOPS + 20) * 16);
    LiveState after;
    CHECK(parse_live_state(r.got[2].payload.data(), 24, &after) && after.flags == LS_REPLY && after.heard == -1 &&
          after.sung == -1 && after.heard_n == 16 && after.hops == TRACK_HOPS + 20);
    // LISTEN on again: everything restarts (seq 0, t_ms from the start, same events as the first time)
    r.got.clear();
    r.ctrl(LC_LISTEN | LC_FAKE);
    r.got.clear();
    for (uint32_t k = 0; k < 100; ++k) r.ch.hop_fake();
    std::vector<uint8_t> seqs2;
    std::vector<LiveNote> again = notes_of(r.got, &seqs2);
    CHECK(!again.empty() && seqs2[0] == 0 && again[0].kind == PIN_CHAIN[0].kind && again[0].t_ms == PIN_CHAIN[0].t_ms);
    bool prefix = again.size() <= n_pin;
    for (size_t i = 0; prefix && i < again.size(); ++i)
      prefix = again[i].kind == PIN_CHAIN[i].kind && again[i].midi == PIN_CHAIN[i].midi && again[i].t_ms == PIN_CHAIN[i].t_ms;
    CHECK(prefix);
    // FAKE off while listening (real mic: here nothing is fed) -> notes released, seq restarts
    r.got.clear();
    r.ctrl(LC_LISTEN);
    std::vector<LiveNote> rel = notes_of(r.got);
    CHECK(r.ch.want_mic() && !r.ch.fake() && r.ch.hops() == 0);
    bool rel_ok = !rel.empty();
    for (const LiveNote& n : rel) rel_ok &= (n.kind == LN_HEARD_OFF || n.kind == LN_SUNG_OFF) && n.flags == LN_FAKE;
    CHECK(rel_ok);
    // real mic in silence: no notes, states with heard -1, flags without LN_FAKE
    r.got.clear();
    int32_t zero[PT_HOP] = {0};
    for (int k = 0; k < 60; ++k) r.ch.hop_raw(zero);
    CHECK(notes_of(r.got).empty() && r.take(T_LIVE_STATE).size() == 10);
  }
  // SING on: the synth gets the same notes, and a LiveVoice renders them
  {
    Rig r;
    r.config(H_THIRD_BELOW, 0);
    r.ctrl(LC_LISTEN | LC_FAKE | LC_SING, 80);
    r.got.clear();
    std::vector<SynthCmd> cmds;
    for (uint32_t k = 0; k < TRACK_HOPS; ++k) {
      r.ch.hop_fake();
      SynthCmd c;
      while (r.v.q.pop(&c)) cmds.push_back(c);
    }
    std::vector<LiveNote> ns = notes_of(r.got);
    std::vector<int> sung_ev, sung_cmd;
    for (const LiveNote& n : ns)
      if (n.kind == LN_SUNG_ON) sung_ev.push_back(n.midi);
    int offs = 0;
    for (const SynthCmd& c : cmds) {
      if (c.op == SC_NOTE_ON) sung_cmd.push_back(c.midi);
      if (c.op == SC_NOTE_OFF) ++offs;
    }
    CHECK(sung_cmd == sung_ev && offs == 15 && sung_ev.size() == 15);
    CHECK(fabsf(r.ch.gain() - 0.35f * 80 / 100) < 1e-6f);
    // the speaker side: start, blocks, stop fade
    LiveVoice& v = r.v;
    v.start();
    std::vector<int16_t> lr(2 * SY_PLAY_BLOCK);
    v.q.push(SynthCmd{SC_NOTE_ON, 57, 100});
    int peak = 0;
    for (int b = 0; b < 10; ++b) {
      v.block(lr.data(), r.ch.gain());
      for (int16_t x : lr) peak = std::max(peak, std::abs((int)x));
    }
    CHECK(v.blocks.load() == 10 && peak > 1000 && peak <= (int)(0.35f * 0.8f * 32767.0f) + 1);
    v.begin_stop();
    int fade = 0;
    while (v.stop_block(lr.data(), r.ch.gain())) ++fade;
    CHECK(fade >= 1 && fade <= 8 && v.synth.idle() && v.blocks.load() == 10u + (uint32_t)fade);
    // SYNTH_NOTE: plays only with SING on, reported with LN_FROM_HOST, does not count as plug-in sung
    r.got.clear();
    r.send(T_SYNTH_NOTE, 0, {69, 90});
    r.send(T_SYNTH_NOTE, 1, {69, 0});
    std::vector<LiveNote> hn = notes_of(r.got);
    CHECK(hn.size() == 2 && hn[0].kind == LN_SUNG_ON && hn[0].midi == 69 && hn[0].vel == 90 &&
          hn[0].flags == LN_FROM_HOST && hn[1].kind == LN_SUNG_OFF && hn[1].flags == LN_FROM_HOST);
    SynthCmd c;
    CHECK(r.v.q.pop(&c) && c.op == SC_NOTE_ON && c.midi == 69 && r.v.q.pop(&c) && c.op == SC_NOTE_OFF);
    r.ctrl(LC_LISTEN | LC_FAKE);  // SING off -> ALL_OFF to the synth
    r.v.drop_pending();
    r.got.clear();
    r.send(T_SYNTH_NOTE, 2, {69, 90});
    CHECK(notes_of(r.got).empty() && !r.v.q.pop(&c));
    // LISTEN off, SING on: SYNTH_NOTE events carry t_ms 0
    r.ctrl(LC_SING);
    r.got.clear();
    r.send(T_SYNTH_NOTE, 3, {60, 50});
    hn = notes_of(r.got);
    CHECK(hn.size() == 1 && hn[0].t_ms == 0 && hn[0].kind == LN_SUNG_ON);
    // volume > 100 is treated as 100
    r.ctrl(LC_SING, 250);
    CHECK(r.ch.volume() == 100 && r.ch.gain() == SY_PLAY_VOLUME);
  }
  // no plug-in configured: HEARD events only, sung = -1
  {
    Rig r;
    r.ctrl(LC_LISTEN | LC_FAKE);
    for (uint32_t k = 0; k < TRACK_HOPS; ++k) r.ch.hop_fake();
    std::vector<LiveNote> ns = notes_of(r.got);
    bool heard_only = ns.size() == 30;
    for (const LiveNote& n : ns) heard_only &= n.kind == LN_HEARD_ON || n.kind == LN_HEARD_OFF;
    CHECK(heard_only);
    CHECK(r.ch.sung() == -1);
  }
  // echo plug-in and a tempo: onset and prev_dur in beats reach the plug-in (echo returns her note)
  {
    Rig r;
    r.config(H_ECHO, 0);
    r.ctrl(LC_LISTEN | LC_FAKE, 100, 60);
    for (uint32_t k = 0; k < 100; ++k) r.ch.hop_fake();
    std::vector<LiveNote> ns = notes_of(r.got);
    bool echo = false;
    for (size_t i = 0; i + 1 < ns.size(); ++i)
      if (ns[i].kind == LN_HEARD_ON && ns[i + 1].kind == LN_SUNG_ON) echo = ns[i + 1].midi == ns[i].midi;
    CHECK(echo);
  }
  // MONITOR: every hop goes out as MIC_HOP (512 bytes through Engine::send), equal to the generator, seq from 0
  {
    Rig r;
    r.ctrl(LC_LISTEN | LC_FAKE | LC_MONITOR);
    r.got.clear();
    for (uint32_t k = 0; k < 300; ++k) r.ch.hop_fake();
    std::vector<Got> hops = r.take(T_MIC_HOP);
    bool ok = hops.size() == 300;
    int16_t want[PT_HOP];
    for (size_t k = 0; ok && k < hops.size(); ++k) {
      fake16_fill((uint32_t)k * PT_HOP, want, PT_HOP);
      ok = hops[k].seq == (uint8_t)k && hops[k].payload.size() == MIC_HOP_LEN;
      for (int i = 0; ok && i < PT_HOP; ++i) ok = (int16_t)get_u16(hops[k].payload.data() + 2 * i) == want[i];
    }
    CHECK(ok);
    // the real-mic path monitors the 24-bit sample scaled to int16
    r.ctrl(LC_LISTEN | LC_MONITOR);
    r.got.clear();
    int32_t raw[PT_HOP];
    for (int i = 0; i < PT_HOP; ++i) raw[i] = (int32_t)((i - 128) * 1000) * 256 + 77;
    r.ch.hop_raw(raw);
    hops = r.take(T_MIC_HOP);
    CHECK(hops.size() == 1 && hops[0].seq == 0 && (int16_t)get_u16(hops[0].payload.data()) == PitchTracker::raw_to_i16(raw[0]));
  }
  // repeatability: two runs give byte-identical output
  {
    std::vector<std::vector<Got>> runs;
    for (int run = 0; run < 2; ++run) {
      Rig r;
      r.config(H_THIRD_BELOW, 0);
      r.ctrl(LC_LISTEN | LC_FAKE | LC_MONITOR | LC_SING);
      for (uint32_t k = 0; k < TRACK_HOPS; ++k) r.ch.hop_fake();
      runs.push_back(r.got);
    }
    bool same = runs[0].size() == runs[1].size() && !runs[0].empty();
    for (size_t i = 0; same && i < runs[0].size(); ++i)
      same = runs[0][i].type == runs[1][i].type && runs[0][i].seq == runs[1][i].seq && runs[0][i].payload == runs[1][i].payload;
    CHECK(same);
  }
  // the old frames still work with the chain attached: PING, NOTE_IN
  {
    Rig r;
    r.config(H_THIRD_BELOW, 0);
    r.ctrl(LC_LISTEN | LC_FAKE);
    r.got.clear();
    r.send(T_PING, 9, {1, 2, 3, 4});
    NoteIn in{0, 64, false, 0.0f, 0.0f};
    std::vector<uint8_t> p(NOTE_IN_LEN);
    pack_note_in(in, p.data());
    r.send(T_NOTE_IN, 10, p);
    CHECK(r.got.size() == 2 && r.got[0].type == T_PONG && r.got[0].seq == 9 && r.got[1].type == T_NOTE_OUT &&
          r.got[1].seq == 10 && (int8_t)r.got[1].payload[2] == 60);
  }
}

// ------------------------------------------------------------------ 8. INFO stays within 240 bytes
static size_t g_info_mode = 0;
static size_t info_fn(char* buf, size_t cap, void*) {
  LiveInfoFacts f;
  if (g_info_mode == 0) {  // the longest possible values
    f = LiveInfoFacts{"ESP32-S3-WROOM-1-N16R8xx", 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u,
                      4294967295u, -2147483647 - 1, 4294967295u, 4294967295u, 4294967295u, 4294967295u,
                      {-128, -128, -128}, {-128, -128, -128}};
  } else {  // what the board will say (ESP32-S3 N16R8, GPIO48 LED)
    f = LiveInfoFacts{"ESP32-S3", 2, 240, 16777216, 8388608, 8300000, 280000, 48, 0, 86400000,
                      16000, 22050, {11, 12, 13}, {4, 5, 6}};
  }
  return live_info(buf, cap, f);
}
static void test_info() {
  for (g_info_mode = 0; g_info_mode < 2; ++g_info_mode) {
    Rig r;
    r.e.begin(Rig::sink, Rig::micros, info_fn, &r, "duet_live", "0.3.0");
    r.send(T_HELLO, 3, {});
    std::vector<Got> info = r.take(T_INFO);
    CHECK(info.size() == 1);
    if (info.empty()) continue;
    std::string text(info[0].payload.begin(), info[0].payload.end());
    if (g_dump) printf("INFO (%s, %zu bytes): %s\n", g_info_mode ? "typical" : "longest", text.size(), text.c_str());
    CHECK(text.size() <= MAX_TEXT);
    // every field is whole key=value; the keys the checks need are all there
    bool whole = true;
    size_t pos = 0;
    std::vector<std::string> keys;
    while (pos <= text.size()) {
      size_t end = text.find(';', pos);
      if (end == std::string::npos) end = text.size();
      const std::string part = text.substr(pos, end - pos);
      const size_t eq = part.find('=');
      whole &= eq != std::string::npos && eq > 0;
      if (eq != std::string::npos) keys.push_back(part.substr(0, eq));
      pos = end + 1;
    }
    CHECK(whole);
    for (const char* k : {"fw", "ver", "proto", "plugins", "flash", "psram", "live", "mic_rate", "spk_rate",
                          "mic_pins", "spk_pins"})
      CHECK(std::find(keys.begin(), keys.end(), std::string(k)) != keys.end());
    if (g_info_mode == 1) {
      CHECK(text.find("fw=duet_live;ver=0.3.0;proto=2;plugins=echo,third_below;flash=16777216;psram=8388608;live=1;"
                      "mic_rate=16000;spk_rate=22050;mic_pins=11/12/13;spk_pins=4/5/6;") == 0);
      for (const char* k : {"chip", "rev", "cpu_mhz", "heap_free", "rgb_gpio", "tx_dropped"})
        CHECK(std::find(keys.begin(), keys.end(), std::string(k)) != keys.end());
    }
  }
}

int main(int argc, char** argv) {
  for (int k = 1; k < argc; ++k) {
    if (strcmp(argv[k], "--dump") == 0) g_dump = true;
  }
  test_constants();
  test_fake16();
  test_yin();
  test_tracker_melody();
  test_synth();
  test_live_protocol();
  test_chain();
  test_info();
  printf("live tests: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
