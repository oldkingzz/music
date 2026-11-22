// The wavetable synth, lifted from the team's sketch (Xinrui Gao, quantization_code_4; see duet_synth.h for the
// credit and the list of changes). The bodies follow their code line for line; "their:" comments quote what was
// replaced.
#include "duet_synth.h"

#include <math.h>

namespace duet {

Synth::Synth(uint32_t rate) : rate_(rate ? rate : SY_PLAY_RATE), fast_release_(false) {
  // their: static const locals of renderVoices(), with PLAY_RATE -> rate_
  attack_step_ = 1.0f / (SY_ATTACK_SEC * (float)rate_);
  decay_factor_ = expf(-1.0f / (SY_DECAY_TAU_SEC * (float)rate_));
  rel_normal_ = expf(-1.0f / (SY_RELEASE_TAU_SEC * (float)rate_));
  rel_fast_ = expf(-1.0f / (SY_FAST_RELEASE_SEC * (float)rate_));
  for (int i = 0; i < SY_NUM_VOICES; i++) voices_[i] = Voice{false, false, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f};
  build_wave();
}

// ---------------- their buildWave ----------------
void Synth::build_wave() {
  const float harm[] = {1.0f, 0.55f, 0.35f, 0.22f, 0.14f, 0.09f, 0.05f, 0.03f};
  float peak = 0.0f;
  for (int i = 0; i < SY_WAVE_N; i++) {
    float x = 2.0f * (float)M_PI * (float)i / (float)SY_WAVE_N;
    float s = 0.0f;
    for (int h = 0; h < 8; h++) s += harm[h] * sinf((float)(h + 1) * x);
    wave_[i] = s;
    if (fabsf(s) > peak) peak = fabsf(s);
  }
  for (int i = 0; i < SY_WAVE_N; i++) wave_[i] /= peak;
  wave_[SY_WAVE_N] = wave_[0];  // guard point for interpolation
}

// ---------------- their voiceNoteOn ----------------
void Synth::note_on(uint8_t note, uint8_t vel) {
  int slot = -1;
  for (int i = 0; i < SY_NUM_VOICES; i++)
    if (!voices_[i].active) {
      slot = i;
      break;
    }
  if (slot < 0)
    for (int i = 0; i < SY_NUM_VOICES; i++)
      if (!voices_[i].gate) {
        slot = i;
        break;
      }
  if (slot < 0) slot = 0;
  float freq = 440.0f * powf(2.0f, ((float)note - 69.0f) / 12.0f);
  Voice& v = voices_[slot];
  v.active = true;
  v.gate = true;
  v.note = note;
  v.phase = 0;
  v.inc = (uint32_t)((double)freq * 4294967296.0 / (double)rate_);  // their: / (double)PLAY_RATE
  v.amp = 0.3f + 0.7f * ((float)vel / 127.0f);
  v.attack = 0.0f;
  v.decay = 1.0f;
  v.rel = 1.0f;
}

// ---------------- their voiceNoteOff ----------------
void Synth::note_off(uint8_t note) {
  for (int i = 0; i < SY_NUM_VOICES; i++) {
    if (voices_[i].active && voices_[i].gate && voices_[i].note == note) {
      voices_[i].gate = false;
      return;
    }
  }
}

// ---------------- their voicesIdle ----------------
bool Synth::idle() const {
  for (int i = 0; i < SY_NUM_VOICES; i++)
    if (voices_[i].active) return false;
  return true;
}

void Synth::reset() {
  for (int i = 0; i < SY_NUM_VOICES; i++) voices_[i].active = false;  // their: before playback starts
  fast_release_ = false;
}

void Synth::release_all() {
  for (int i = 0; i < SY_NUM_VOICES; i++) voices_[i].gate = false;
}

void Synth::release_all_fast() {
  for (int i = 0; i < SY_NUM_VOICES; i++) voices_[i].gate = false;  // their: if (playStop) { ... }
  fast_release_ = true;
}

// ---------------- their renderVoices ----------------
float Synth::render_sample() {
  const uint32_t FRAC_BITS = 32 - SY_WAVE_BITS;
  const uint32_t FRAC_MASK = (1u << FRAC_BITS) - 1u;
  const float FRAC_SCALE = 1.0f / (float)(1u << FRAC_BITS);

  float acc = 0.0f;
  for (int k = 0; k < SY_NUM_VOICES; k++) {
    Voice& v = voices_[k];
    if (!v.active) continue;

    uint32_t idx = v.phase >> FRAC_BITS;
    float frac = (float)(v.phase & FRAC_MASK) * FRAC_SCALE;
    float s = wave_[idx] + (wave_[idx + 1] - wave_[idx]) * frac;
    v.phase += v.inc;

    if (v.attack < 1.0f) {
      v.attack += attack_step_;
      if (v.attack > 1.0f) v.attack = 1.0f;
    }
    if (v.gate) {
      if (v.decay > SY_DECAY_FLOOR) v.decay *= decay_factor_;
    } else {
      v.rel *= fast_release_ ? rel_fast_ : rel_normal_;
      if (v.rel < 0.0005f) {
        v.active = false;
        continue;
      }
    }
    acc += s * v.attack * v.decay * v.rel * v.amp;
  }
  return acc;
}

// their playback loop, per sample: s = renderVoices() * PLAY_VOLUME; clip; v16 = (int16_t)(s * 32767.0f); L = R = v16
void Synth::render(int16_t* out, size_t frames, bool stereo, float gain) {
  for (size_t i = 0; i < frames; i++) {
    float s = render_sample() * gain;
    if (s > 1.0f) s = 1.0f;
    else if (s < -1.0f) s = -1.0f;
    int16_t v16 = (int16_t)(s * 32767.0f);
    if (stereo) {
      out[2 * i] = v16;
      out[2 * i + 1] = v16;
    } else {
      out[i] = v16;
    }
  }
}

float synth_gain(uint8_t volume_percent) {
  const unsigned v = volume_percent > 100 ? 100u : volume_percent;
  return SY_PLAY_VOLUME * (float)v / 100.0f;
}

void synth_apply(Synth& s, const SynthCmd& c) {
  switch (c.op) {
    case SC_NOTE_ON:
      if (c.vel) s.note_on(c.midi, c.vel);
      else s.note_off(c.midi);
      break;
    case SC_NOTE_OFF: s.note_off(c.midi); break;
    case SC_ALL_OFF: s.release_all(); break;
    default: break;
  }
}

}  // namespace duet
