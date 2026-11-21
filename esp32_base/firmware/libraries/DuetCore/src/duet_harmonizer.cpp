#include "duet_harmonizer.h"

namespace duet {

namespace {

// Python's % for a positive modulus: the result is never negative.
inline int pmod(int a, int m) {
  int r = a % m;
  return r < 0 ? r + m : r;
}

const int MAJOR[7] = {0, 2, 4, 5, 7, 9, 11};  // scale degrees, semitones above the tonic

class Echo : public Harmonizer {
 public:
  const char* name() const override { return "echo"; }
  void start(const Meta&) override {}
  int step(int, float, int midi, float) override { return midi; }
};

// Port of duet_v0/futures/v3_nolatency/systems/third_below.py, line for line.
class ThirdBelow : public Harmonizer {
 public:
  const char* name() const override { return "third_below"; }
  void start(const Meta& meta) override { t_ = meta.tonic; }
  int step(int, float, int midi, float) override {
    const int pc = pmod(midi - t_, 12);
    int j = -1;
    for (int k = 0; k < 7; ++k) {
      if (MAJOR[k] == pc) j = k;
    }
    int h;
    if (j >= 0) {
      h = midi - pmod(pc - MAJOR[pmod(j - 2, 7)], 12);  // two scale steps down
    } else {
      h = midi - 3;
    }
    while (h > ALTO_HI) h -= 12;
    while (h < ALTO_LO) h += 12;
    return h;
  }

 private:
  int t_ = 0;
};

Echo g_echo;
ThirdBelow g_third_below;

}  // namespace

Harmonizer* harmonizer(uint8_t id) {
  switch (id) {
    case H_ECHO: return &g_echo;
    case H_THIRD_BELOW: return &g_third_below;
    default: return nullptr;
  }
}

}  // namespace duet
