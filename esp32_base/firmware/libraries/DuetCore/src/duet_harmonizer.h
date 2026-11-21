// The plug-in slot: one harmony model = one Harmonizer. Portable C++.
//
// The interface is the one every model scored by bench3 already has in Python
// (duet_v0/futures/v3_nolatency/bench3.py):
//
//     obj.start(meta)                           meta = {'tonic', 'M', 'K', 'pos'}
//     obj.step(i, onset, midi, prev_dur) -> Duet's MIDI note, or -1 = silent
//
// so a model that has a bench3 score has a defined port target here and a reference output to compare with.
// What a plug-in may know at her note i: pitches and onsets of notes 0..i, lengths of notes 0..i-1, the key,
// the meter and the beat grid. Nothing from the future.
//
// Adding a model: write a class in duet_harmonizer.cpp (or its own .cpp), give it the next id below, add it
// to the table in harmonizer(). Then add the same id to host/duet_host/harmonizers.py with the Python reference.
#pragma once
#include <stdint.h>

namespace duet {

static const int ALTO_LO = 53;  // F3
static const int ALTO_HI = 77;  // F5, the range Duet may sing in (duet_beat/theory.py ALTO)

struct Meta {
  int tonic;           // home note of the key (any MIDI note of that pitch class; use it mod 12)
  int M;               // beats per bar
  int K;               // bench3's K, passed through
  const uint8_t* pos;  // bar position of each beat (0 = downbeat); valid until the next start()
  int n_pos;
};

class Harmonizer {
 public:
  virtual ~Harmonizer() {}
  virtual const char* name() const = 0;
  virtual void start(const Meta& meta) = 0;
  // i: index of her note. onset: beats. midi: her pitch. prev_dur: length of her previous note in beats,
  // negative when unknown (first note). Returns Duet's MIDI note, or -1 to stay silent.
  virtual int step(int i, float onset, int midi, float prev_dur) = 0;
};

enum : uint8_t {
  H_ECHO = 0,         // answers her own note: tests the link and the note path, not music
  H_THIRD_BELOW = 1,  // two scale steps below her note, kept inside the alto range (bench3 third_below)
  H_COUNT = 2,
};

// The plug-in with this id, or nullptr. Instances are static: no heap, one song at a time.
Harmonizer* harmonizer(uint8_t id);

}  // namespace duet
