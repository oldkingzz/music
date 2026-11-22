#include "duet_live.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

namespace duet {

// ------------------------------------------------------------------ LiveVoice (speaker side)
void LiveVoice::block(int16_t* lr, float gain) {
  SynthCmd c;
  while (q.pop(&c)) synth_apply(synth, c);
  synth.render(lr, SY_PLAY_BLOCK, true, gain);
  blocks.fetch_add(1, std::memory_order_relaxed);
}

void LiveVoice::begin_stop() {
  SynthCmd c;
  while (q.pop(&c)) synth_apply(synth, c);
  synth.release_all_fast();
  stop_left_ = SY_STOP_FADE_BLOCKS;
}

// their: for (int blk = 0; blk < 30 && !voicesIdle() && ok; blk++) { render a block; write it }
bool LiveVoice::stop_block(int16_t* lr, float gain) {
  if (stop_left_ <= 0 || synth.idle()) {
    stop_left_ = 0;
    return false;
  }
  --stop_left_;
  synth.render(lr, SY_PLAY_BLOCK, true, gain);
  blocks.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void LiveVoice::drop_pending() {
  SynthCmd c;
  while (q.pop(&c)) {
  }
}

// ------------------------------------------------------------------ LiveChain
LiveChain::LiveChain()
    : engine_(nullptr), voice_(nullptr), hw_(nullptr), hw_ctx_(nullptr), flags_(0), volume_(100), epoch_(0),
      bpm_(LIVE_DEFAULT_BPM), fake_gap_(FAKE16_GAP) {
  restart_source();
  epoch_.store(0);
}

void LiveChain::begin(Engine* engine, LiveVoice* voice, LiveHwFn hw, void* hw_ctx) {
  engine_ = engine;
  voice_ = voice;
  hw_ = hw;
  hw_ctx_ = hw_ctx;
}

bool LiveChain::on_frame(const Frame& f) {
  if (f.type == T_LIVE_CTRL) {
    LiveCtrl c;
    if (parse_live_ctrl(f.payload, f.len, &c)) control(c, f.seq);
    return true;
  }
  if (f.type == T_SYNTH_NOTE) {
    if (f.len == SYNTH_NOTE_LEN) synth_note(f.payload[0], f.payload[1]);
    return true;
  }
  return false;
}

void LiveChain::restart_source() {
  tracker_.reset();
  fake_pos_ = 0;
  hops_ = 0;
  heard_n_ = sung_n_ = 0;
  note_seq_ = state_seq_ = hop_seq_ = 0;
  heard_i_ = 0;
  heard_on_ms_ = 0;
  have_prev_ = false;
  prev_dur_ms_ = 0;
  sung_ = -1;
  freq_ = 0.0f;
  conf_ = 0;
  level_ = -100;
  cents_ = 0;
  epoch_.fetch_add(1);
}

void LiveChain::release_all(uint8_t old_flags) {
  const uint8_t src = (old_flags & LC_FAKE) ? LN_FAKE : 0;
  const uint32_t t = hops_ * LIVE_HOP_MS;
  if (tracker_.active() >= 0) emit(LN_HEARD_OFF, (uint8_t)tracker_.active(), 0, src, t);
  end_sung(src, t);
  tracker_.reset();  // heard = -1 from now on
}

void LiveChain::control(const LiveCtrl& c, int reply_seq) {
  const uint8_t old = flags_.load();
  const uint8_t nf = c.flags & LC_KNOWN;
  const bool was_listen = (old & LC_LISTEN) != 0, listen = (nf & LC_LISTEN) != 0;
  const bool fake_changed = ((old ^ nf) & LC_FAKE) != 0;
  // Sounding notes end (with their OFF events) before the source stops or changes.
  if (was_listen && (!listen || fake_changed)) release_all(old);
  if ((old & LC_SING) && !(nf & LC_SING)) synth_cmd(SC_ALL_OFF, 0, 0);
  volume_.store(c.volume > 100 ? 100 : c.volume);
  bpm_ = c.bpm ? c.bpm : LIVE_DEFAULT_BPM;
  flags_.store(nf);
  if (listen && (!was_listen || fake_changed)) restart_source();
  if (reply_seq >= 0) send_state((uint8_t)reply_seq, true);
}

void LiveChain::synth_note(uint8_t midi, uint8_t vel) {
  if (!singing() || midi > 127) return;
  if (vel) {
    synth_cmd(SC_NOTE_ON, midi, vel);
    emit(LN_SUNG_ON, midi, vel, LN_FROM_HOST, now_ms());
  } else {
    synth_cmd(SC_NOTE_OFF, midi, 0);
    emit(LN_SUNG_OFF, midi, 0, LN_FROM_HOST, now_ms());
  }
}

void LiveChain::hop_fake() {
  fake16_fill(fake_pos_, x16_, PT_HOP, fake_gap_);
  fake_pos_ += PT_HOP;
  PitchEvent ev[PitchTracker::MAX_EVENTS];
  const int n = tracker_.hop_i16(x16_, ev);
  after_hop(x16_, n, ev);
}

void LiveChain::hop_raw(const int32_t* raw) {
  for (int i = 0; i < PT_HOP; i++) x16_[i] = PitchTracker::raw_to_i16(raw[i]);
  PitchEvent ev[PitchTracker::MAX_EVENTS];
  const int n = tracker_.hop_raw(raw, ev);
  after_hop(x16_, n, ev);
}

void LiveChain::after_hop(const int16_t* x16, int n_ev, const PitchEvent* ev) {
  if (flags_.load() & LC_MONITOR) {
    for (int i = 0; i < PT_HOP; i++) put_u16(hop_payload_ + 2 * i, (uint16_t)x16[i]);
    if (engine_) engine_->send(T_MIC_HOP, hop_seq_, hop_payload_, MIC_HOP_LEN);
    ++hop_seq_;
  }
  ++hops_;
  const uint32_t t = hops_ * LIVE_HOP_MS;
  const uint8_t src = fake() ? LN_FAKE : 0;
  for (int k = 0; k < n_ev; ++k) {
    const PitchEvent& e = ev[k];
    if (e.kind == PT_NOTE_OFF) {
      emit(LN_HEARD_OFF, e.midi, 0, src, t);
      prev_dur_ms_ = t - heard_on_ms_;
      have_prev_ = true;
      end_sung(src, t);
    } else if (e.kind == PT_NOTE_ON) {
      emit(LN_HEARD_ON, e.midi, e.vel, src, t);
      ++heard_n_;
      end_sung(src, t);  // her next note replaces Duet's (normally her OFF came first and ended it)
      const int i = heard_i_++;
      const float onset = beats(t);
      const float prev = have_prev_ ? beats(prev_dur_ms_) : -1.0f;
      heard_on_ms_ = t;
      const int m = (engine_ && engine_->plugin()) ? engine_->plugin_step(i, onset, e.midi, prev) : -1;
      if (m >= 0) {
        sung_ = m;
        ++sung_n_;
        emit(LN_SUNG_ON, (uint8_t)m, e.vel, src, t);
        synth_cmd(SC_NOTE_ON, (uint8_t)m, e.vel);
      }
    }
  }
  // latest-hop facts for LIVE_STATE
  freq_ = tracker_.freq();
  conf_ = (uint8_t)pt_constrain((long)lroundf(tracker_.conf() * 100.0f), 0L, 100L);
  const float rms = tracker_.rms();
  float db = rms > 0.0f ? 20.0f * log10f(rms) : -100.0f;
  db = pt_constrain(db, -100.0f, 0.0f);  // also catches -inf; a NaN rms stays at -100 below
  level_ = (int8_t)(db == db ? lroundf(db) : -100);
  cents_ = (int8_t)pt_constrain((long)lroundf(tracker_.cents()), -50L, 50L);
  if (hops_ % LIVE_STATE_EVERY == 0) send_state(state_seq_++, false);
}

void LiveChain::end_sung(uint8_t src, uint32_t t_ms) {
  if (sung_ < 0) return;
  emit(LN_SUNG_OFF, (uint8_t)sung_, 0, src, t_ms);
  synth_cmd(SC_NOTE_OFF, (uint8_t)sung_, 0);
  sung_ = -1;
}

void LiveChain::emit(uint8_t kind, uint8_t midi, uint8_t vel, uint8_t fl, uint32_t t_ms) {
  LiveNote n{kind, midi, vel, fl, t_ms};
  uint8_t p[LIVE_NOTE_LEN];
  pack_live_note(n, p);
  if (engine_) engine_->send(T_LIVE_NOTE, note_seq_, p, LIVE_NOTE_LEN);
  ++note_seq_;
}

void LiveChain::synth_cmd(uint8_t op, uint8_t midi, uint8_t vel) {
  // Only while SING is on: with the speaker off nothing renders, and the speaker side drops what is queued.
  if (!voice_ || !(flags_.load() & LC_SING)) return;
  voice_->q.push(SynthCmd{op, midi, vel});
}

void LiveChain::send_state(uint8_t seq, bool reply) {
  LiveHw hw{false, false, false};
  if (hw_) hw_(&hw, hw_ctx_);
  LiveState s;
  s.flags = (uint8_t)((flags_.load() & LC_KNOWN) | (hw.mic_up ? LS_MIC_UP : 0) | (hw.spk_up ? LS_SPK_UP : 0) |
                      (hw.error ? LS_ERROR : 0) | (reply ? LS_REPLY : 0));
  s.volume = volume_.load();
  s.heard = (int8_t)tracker_.active();
  s.sung = (int8_t)sung_;
  s.freq = freq_;
  s.conf = conf_;
  s.level = level_;
  s.cents = cents_;
  s.hops = hops_;
  s.heard_n = heard_n_;
  s.sung_n = sung_n_;
  s.spk_blocks = voice_ ? voice_->blocks.load(std::memory_order_relaxed) : 0;
  uint8_t p[LIVE_STATE_LEN];
  pack_live_state(s, p);
  if (engine_) engine_->send(T_LIVE_STATE, seq, p, LIVE_STATE_LEN);
}

// ------------------------------------------------------------------ INFO
size_t info_append(char* buf, size_t cap, size_t n, const char* text) {
  const size_t k = strlen(text);
  if (n + k + 1 > cap) return n;  // +1 for the NUL
  memcpy(buf + n, text, k + 1);
  return n + k;
}

size_t live_info_head(char* buf, size_t cap, const LiveInfoFacts& f) {
  if (cap == 0) return 0;
  char tmp[80];
  buf[0] = 0;
  size_t n = 0;
  snprintf(tmp, sizeof(tmp), "flash=%lu;psram=%lu", (unsigned long)f.flash, (unsigned long)f.psram);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";live=1;mic_rate=%lu;spk_rate=%lu", (unsigned long)f.mic_rate, (unsigned long)f.spk_rate);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";mic_pins=%d/%d/%d;spk_pins=%d/%d/%d", f.mic_pins[0], f.mic_pins[1], f.mic_pins[2],
           f.spk_pins[0], f.spk_pins[1], f.spk_pins[2]);
  n = info_append(buf, cap, n, tmp);
  return n;
}

size_t live_info(char* buf, size_t cap, const LiveInfoFacts& f) {
  if (cap == 0) return 0;
  char tmp[80];
  size_t n = live_info_head(buf, cap, f);
  snprintf(tmp, sizeof(tmp), ";chip=%.20s;rev=%u;cpu_mhz=%u", f.chip ? f.chip : "?", f.rev, f.cpu_mhz);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";heap_free=%lu;rgb_gpio=%d", (unsigned long)f.heap_free, f.rgb_gpio);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";tx_dropped=%lu", (unsigned long)f.tx_dropped);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";psram_free=%lu", (unsigned long)f.psram_free);
  n = info_append(buf, cap, n, tmp);
  snprintf(tmp, sizeof(tmp), ";uptime_ms=%lu", (unsigned long)f.uptime_ms);
  n = info_append(buf, cap, n, tmp);
  return n;
}

}  // namespace duet
