// Payloads of the note path: CONFIG, CONFIG_ACK, NOTE_IN, NOTE_OUT. Header-only, portable.
// Floats on the wire are IEEE-754 single precision, little-endian (the ESP32-S3 and the Mac both are).
#pragma once
#include <stdint.h>
#include <string.h>

#include "duet_protocol.h"

namespace duet {

// CONFIG: which plug-in to run and what it is told about the song before the first note.
// Mirrors bench3's start(meta): meta = {'tonic', 'M', 'K', 'pos'}.
struct ConfigMsg {
  uint8_t plugin;      // plug-in id (duet_harmonizer.h)
  uint8_t tonic;       // home note of the key: pitch class 0..11 (C = 0) or any MIDI note of it; used mod 12
  uint8_t M;           // beats per bar
  uint8_t flags;       // reserved, send 0
  uint16_t K;          // bench3's K, passed through
  uint16_t n_pos;      // length of pos
  const uint8_t* pos;  // bar position of each beat (0 = downbeat), n_pos entries, points into the frame
};

static inline bool parse_config(const uint8_t* p, uint16_t len, ConfigMsg* c) {
  if (len < CONFIG_HEADER_LEN) return false;
  c->plugin = p[0];
  c->tonic = p[1];
  c->M = p[2];
  c->flags = p[3];
  c->K = get_u16(p + 4);
  c->n_pos = get_u16(p + 6);
  if ((size_t)CONFIG_HEADER_LEN + c->n_pos != len || c->n_pos > MAX_POS) return false;
  c->pos = p + CONFIG_HEADER_LEN;
  return true;
}

enum : uint8_t {
  ACK_OK = 0,
  ACK_UNKNOWN_PLUGIN = 1,
  ACK_BAD_CONFIG = 2,
};

// NOTE_IN: one of her notes, sent at its onset.
struct NoteIn {
  uint16_t i;      // index of her note, 0 for the first note after CONFIG
  uint8_t midi;    // her pitch (MIDI note number)
  bool has_prev;   // false for the first note: the length of the previous note is not known
  float onset;     // beats from the start of the song
  float prev_dur;  // length of her previous note in beats (only if has_prev)
};

static inline bool parse_note_in(const uint8_t* p, uint16_t len, NoteIn* n) {
  if (len != NOTE_IN_LEN) return false;
  n->i = get_u16(p);
  n->midi = p[2];
  n->has_prev = (p[3] & 1) != 0;
  memcpy(&n->onset, p + 4, 4);
  memcpy(&n->prev_dur, p + 8, 4);
  return true;
}

static inline void pack_note_in(const NoteIn& n, uint8_t* p) {
  put_u16(p, n.i);
  p[2] = n.midi;
  p[3] = n.has_prev ? 1 : 0;
  memcpy(p + 4, &n.onset, 4);
  memcpy(p + 8, &n.prev_dur, 4);
}

enum : uint8_t {
  NOTE_OK = 0,
  NOTE_NOT_CONFIGURED = 1,  // NOTE_IN before a successful CONFIG: the answer is silence
};

// NOTE_OUT: Duet's answer to NOTE_IN number i. Its frame seq equals the seq of that NOTE_IN.
struct NoteOut {
  uint16_t i;
  int8_t midi;          // Duet's note, or -1 = stay silent
  uint8_t status;       // NOTE_OK, ...
  uint32_t compute_us;  // time spent inside the plug-in's step(), measured on the device
};

static inline void pack_note_out(const NoteOut& n, uint8_t* p) {
  put_u16(p, n.i);
  p[2] = (uint8_t)n.midi;
  p[3] = n.status;
  put_u32(p + 4, n.compute_us);
}

static inline bool parse_note_out(const uint8_t* p, uint16_t len, NoteOut* n) {
  if (len != NOTE_OUT_LEN) return false;
  n->i = get_u16(p);
  n->midi = (int8_t)p[2];
  n->status = p[3];
  n->compute_us = get_u32(p + 4);
  return true;
}

}  // namespace duet
