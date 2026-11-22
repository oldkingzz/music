// YIN pitch and the note decision, lifted from the team's sketch (Christine, quantization_code_3 / _4; see
// duet_pitch.h for the credit and the list of changes). The bodies below follow their code line for line;
// comments starting with "their:" quote what was replaced.
#include "duet_pitch.h"

#include <math.h>
#include <string.h>

namespace duet {

void PitchTracker::reset() {
  memset(frame_, 0, sizeof(frame_));
  memset(d_, 0, sizeof(d_));
  dc_ = 0.0f;
  active_ = -1;
  hist_count_ = 0;
  unvoiced_run_ = 0;
  for (int i = 0; i < PT_SETTLE_FRAMES; i++) {
    pitch_hist_[i] = 0;
    rms_hist_[i] = 0;
  }
  ui_freq_ = ui_cents_ = ui_conf_ = rms_ = midi_f_ = 0.0f;
  voiced_ = false;
}

int16_t PitchTracker::raw_to_i16(int32_t raw) {
  const int32_t s24 = raw >> 8;           // the 24-bit sample, as their scaling takes it
  int32_t v = (s24 + 128) >> 8;           // to 16 bits, rounded
  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;
  return (int16_t)v;
}

// ---------------- YIN pitch detector (their yinPitch) ----------------
float PitchTracker::yin(const float* x, float& confidence) {
  float* d = d_;  // their: static float d[TAU_MAX + 2];
  d[0] = 1.0f;
  float running = 0.0f;

  for (int tau = 1; tau <= PT_TAU_MAX; tau++) {
    float sum = 0.0f;
    for (int j = 0; j < PT_W; j++) {
      float diff = x[j] - x[j + tau];
      sum += diff * diff;
    }
    running += sum;
    d[tau] = (running > 0.0f) ? (sum * tau / running) : 1.0f;
  }

  int tauEst = -1;
  for (int tau = PT_TAU_MIN; tau <= PT_TAU_MAX; tau++) {
    if (d[tau] < PT_YIN_THRESHOLD) {
      while (tau + 1 <= PT_TAU_MAX && d[tau + 1] < d[tau]) tau++;
      tauEst = tau;
      break;
    }
  }
  if (tauEst < 0) {
    float best = 1e9f;
    for (int tau = PT_TAU_MIN; tau <= PT_TAU_MAX; tau++) {
      if (d[tau] < best) {
        best = d[tau];
        tauEst = tau;
      }
    }
  }
  if (tauEst < 1) {
    confidence = 0;
    return 0;
  }

  confidence = pt_constrain(1.0f - d[tauEst], 0.0f, 1.0f);  // their: constrain(1.0f - d[tauEst], 0.0f, 1.0f)

  float tauF = (float)tauEst;
  if (tauEst > 1 && tauEst < PT_TAU_MAX) {
    float s0 = d[tauEst - 1], s1 = d[tauEst], s2 = d[tauEst + 1];
    float denom = 2.0f * (2.0f * s1 - s2 - s0);
    if (fabsf(denom) > 1e-9f) tauF += (s2 - s0) / denom;
  }
  return (float)PT_SAMPLE_RATE / tauF;
}

// ---------------- one hop of their audioTask ----------------
int PitchTracker::hop_raw(const int32_t* raw, PitchEvent* ev) {
  // their: size_t got = i2s.readBytes((char*)raw, sizeof raw); if (got < sizeof raw) continue;  (the caller reads)
  memmove(frame_, frame_ + PT_HOP, (PT_FRAME - PT_HOP) * sizeof(float));
  for (int i = 0; i < PT_HOP; i++) {
    float s = (float)(raw[i] >> 8) / 8388608.0f;
    dc_ += 0.001f * (s - dc_);
    frame_[PT_FRAME - PT_HOP + i] = s - dc_;
  }
  return process(ev);
}

int PitchTracker::hop_i16(const int16_t* x, PitchEvent* ev) {
  int32_t raw[PT_HOP];
  for (int i = 0; i < PT_HOP; i++) raw[i] = (int32_t)x[i] * 65536;  // x << 16 without shifting a negative
  return hop_raw(raw, ev);
}

int PitchTracker::process(PitchEvent* ev) {
  int n_ev = 0;
  // their midiNoteOff(): a note-off is sent only while a note is sounding (midiSounding), i.e. activeNote >= 0.
  // their midiNoteOn(n, vel): a note-on.
  float* frame = frame_;

  float e = 0;
  for (int i = 0; i < PT_FRAME; i++) e += frame[i] * frame[i];
  float rms = sqrtf(e / PT_FRAME);

  // 1) pitch for this frame
  float conf = 0, freq = 0, midiF = 0;
  bool voiced = false;
  if (rms >= PT_RMS_GATE) {  // their: if (rms >= RMS_GATE && playState == PLAY_IDLE)  -- playback gate removed
    float f = yin(frame, conf);
    if (f > 0 && conf >= PT_MIN_CONFIDENCE) {
      float m = 69.0f + 12.0f * log2f(f / 440.0f);
      if (m >= 0.0f && m <= 127.0f) {
        voiced = true;
        freq = f;
        midiF = m;
      }
    }
  } else {
    conf = 0;
  }

  // 2) note decision
  if (voiced) {
    unvoiced_run_ = 0;

    // push this reading into the history (newest at the end)
    if (hist_count_ < PT_SETTLE_FRAMES) hist_count_++;
    for (int i = 0; i < PT_SETTLE_FRAMES - 1; i++) {
      pitch_hist_[i] = pitch_hist_[i + 1];
      rms_hist_[i] = rms_hist_[i + 1];
    }
    pitch_hist_[PT_SETTLE_FRAMES - 1] = midiF;
    rms_hist_[PT_SETTLE_FRAMES - 1] = rms;

    if (hist_count_ >= PT_SETTLE_FRAMES) {
      float tmp[PT_SETTLE_FRAMES];
      float lo = 1e9f, hi = -1e9f, peak = 0.0f;
      for (int i = 0; i < PT_SETTLE_FRAMES; i++) {
        tmp[i] = pitch_hist_[i];
        if (tmp[i] < lo) lo = tmp[i];
        if (tmp[i] > hi) hi = tmp[i];
        if (rms_hist_[i] > peak) peak = rms_hist_[i];
      }

      // only act once the pitch has stopped sliding
      if (hi - lo <= PT_SETTLE_SPREAD) {
        for (int i = 1; i < PT_SETTLE_FRAMES; i++) {  // insertion sort for the median
          float v = tmp[i];
          int j = i - 1;
          while (j >= 0 && tmp[j] > v) {
            tmp[j + 1] = tmp[j];
            j--;
          }
          tmp[j + 1] = v;
        }
        float med = tmp[PT_SETTLE_FRAMES / 2];

        int target = (active_ >= 0 && fabsf(med - active_) < PT_HYSTERESIS) ? active_ : (int)lroundf(med);

        if (target != active_) {
          if (active_ >= 0) ev[n_ev++] = PitchEvent{PT_NOTE_OFF, (uint8_t)active_, 0};  // their: midiNoteOff();
          float db = 20.0f * log10f(peak + 1e-9f);
          // their: constrain((int)lroundf(20.0f + (db + 48.0f) * (107.0f / 30.0f)), 20, 127)
          int vel = pt_constrain((int)lroundf(20.0f + (db + 48.0f) * (107.0f / 30.0f)), 20, 127);
          ev[n_ev++] = PitchEvent{PT_NOTE_ON, (uint8_t)target, (uint8_t)vel};  // their: midiNoteOn(target, vel);
          active_ = target;
        }
      }
    }
  } else {
    hist_count_ = 0;
    if (unvoiced_run_ < 1000) unvoiced_run_++;
    if (active_ >= 0 && unvoiced_run_ >= PT_RELEASE_FRAMES) {
      ev[n_ev++] = PitchEvent{PT_NOTE_OFF, (uint8_t)active_, 0};  // their: midiNoteOff();
      active_ = -1;
    }
  }

  // their UiState
  ui_freq_ = voiced ? freq : 0;
  // their: constrain((midiF - activeNote) * 100.0f, -50.0f, 50.0f)
  ui_cents_ = (active_ >= 0 && voiced) ? pt_constrain((midiF - active_) * 100.0f, -50.0f, 50.0f) : 0;
  ui_conf_ = conf;
  rms_ = rms;
  voiced_ = voiced;
  midi_f_ = midiF;
  return n_ev;
}

}  // namespace duet
