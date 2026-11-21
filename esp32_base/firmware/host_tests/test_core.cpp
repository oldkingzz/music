// Host unit tests of DuetCore (protocol, engine, plug-ins). Built with sanitizers by build_and_test.sh.
//   test_core                       unit tests only
//   test_core <note_vectors.txt>    also replays every song of the vector file through the plug-in named in
//                                   its header and compares each output note with the Python reference
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "DuetCore.h"

using namespace duet;

static int g_checks = 0, g_fail = 0;
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

// ------------------------------------------------------------------ protocol
static void test_protocol() {
  // Every type round-trips at its minimum and maximum length, fed whole and byte by byte.
  const uint8_t types[] = {T_MIC_AUDIO, T_SPK_AUDIO, T_SERVO, T_LED, T_PING, T_STATUS, T_DIAG, T_PONG,
                           T_HELLO, T_CONFIG, T_NOTE_IN, T_AUDIO_CTRL, T_INFO, T_NOTE_OUT, T_LOG, T_CONFIG_ACK};
  for (uint8_t t : types) {
    uint16_t lo = 0, hi = 0;
    CHECK(len_range(t, &lo, &hi));
    for (uint16_t len : {lo, hi}) {
      std::vector<uint8_t> payload(len);
      for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)(i * 7 + t);
      std::vector<uint8_t> fr = frame(t, 42, payload);
      for (int mode = 0; mode < 2; ++mode) {
        Parser p;
        std::vector<Got> got;
        if (mode == 0) {
          p.feed(fr.data(), fr.size(), collect, &got);
        } else {
          for (uint8_t b : fr) p.feed(&b, 1, collect, &got);
        }
        CHECK(got.size() == 1);
        CHECK(!got.empty() && got[0].type == t && got[0].seq == 42 && got[0].payload == payload);
        CHECK(p.pending() == 0 && p.stats().frames_ok == 1 && p.stats().bytes_skipped == 0);
      }
    }
    // One byte outside the range is rejected.
    if (hi < MAX_PAYLOAD) {
      std::vector<uint8_t> fr = frame(t, 1, std::vector<uint8_t>(hi + 1, 0));
      Parser p;
      std::vector<Got> got;
      p.feed(fr.data(), fr.size(), collect, &got);
      CHECK(got.empty() && p.stats().bad_header >= 1);
    }
    if (lo > 0) {
      std::vector<uint8_t> fr = frame(t, 1, std::vector<uint8_t>(lo - 1, 0));
      Parser p;
      std::vector<Got> got;
      p.feed(fr.data(), fr.size(), collect, &got);
      CHECK(got.empty() && p.stats().bad_header >= 1);
    }
  }
  CHECK(!len_range(0x7F, nullptr, nullptr));

  // Version-1 lengths did not change.
  uint16_t lo, hi;
  CHECK(len_range(T_MIC_AUDIO, &lo, &hi) && lo == 480 && hi == 480);
  CHECK(len_range(T_STATUS, &lo, &hi) && lo == 5 && hi == 5);
  CHECK(len_range(T_PONG, &lo, &hi) && lo == 8 && hi == 8);

  // Known bytes: PING seq 7, host time 0x11223344.
  {
    std::vector<uint8_t> fr = frame(T_PING, 7, {0x44, 0x33, 0x22, 0x11});
    const uint8_t want[] = {0xA5, 0x5A, 0x12, 0x07, 0x04, 0x00, 0x44, 0x33, 0x22, 0x11,
                            (uint8_t)(0x12 + 0x07 + 0x04 + 0x44 + 0x33 + 0x22 + 0x11)};
    CHECK(fr.size() == sizeof(want) && memcmp(fr.data(), want, sizeof(want)) == 0);
  }

  // Garbage (boot text, a lone A5, a false sync) before and between frames is skipped.
  {
    std::vector<uint8_t> s;
    const char* boot = "ESP-ROM:esp32s3-20210327\r\nBuild:Mar 27 2021\r\n";
    s.insert(s.end(), boot, boot + strlen(boot));
    s.push_back(0xA5);
    s.push_back(0x00);
    s.push_back(0xA5);
    s.push_back(0x5A);
    s.push_back(0x7F);  // unknown type after a real sync
    std::vector<uint8_t> a = frame(T_PING, 1, {1, 2, 3, 4});
    std::vector<uint8_t> b = frame(T_LED, 2, {2});
    s.insert(s.end(), a.begin(), a.end());
    s.push_back(0x55);
    s.insert(s.end(), b.begin(), b.end());
    Parser p;
    std::vector<Got> got;
    p.feed(s.data(), s.size(), collect, &got);
    CHECK(got.size() == 2);
    CHECK(got.size() == 2 && got[0].type == T_PING && got[1].type == T_LED);
    CHECK(p.stats().bad_header >= 1 && p.stats().bytes_skipped > 0);
  }

  // A corrupted checksum loses that frame only.
  {
    std::vector<uint8_t> a = frame(T_PING, 1, {1, 2, 3, 4});
    std::vector<uint8_t> b = frame(T_PING, 2, {5, 6, 7, 8});
    a[7] ^= 0x10;
    std::vector<uint8_t> s = a;
    s.insert(s.end(), b.begin(), b.end());
    Parser p;
    std::vector<Got> got;
    p.feed(s.data(), s.size(), collect, &got);
    CHECK(got.size() == 1 && got[0].seq == 2);
    CHECK(p.stats().bad_checksum == 1);
  }

  // A header that promises 200 bytes which never come: the frame after it waits until drop_stale().
  {
    std::vector<uint8_t> s = {0xA5, 0x5A, T_LOG, 0, 200, 0};
    std::vector<uint8_t> b = frame(T_PING, 9, {1, 2, 3, 4});
    s.insert(s.end(), b.begin(), b.end());
    Parser p;
    std::vector<Got> got;
    p.feed(s.data(), s.size(), collect, &got);
    CHECK(got.empty() && p.pending() == s.size());
    p.drop_stale(collect, &got);
    CHECK(got.size() == 1 && got[0].type == T_PING && got[0].seq == 9);
    CHECK(p.pending() == 0 && p.stats().stale_drops == 1);
  }

  // A long stream of back-to-back frames in awkward chunk sizes.
  {
    std::vector<uint8_t> s;
    for (int i = 0; i < 300; ++i) {
      std::vector<uint8_t> f = (i % 3 == 0) ? frame(T_MIC_AUDIO, (uint8_t)i, std::vector<uint8_t>(480, (uint8_t)i))
                               : (i % 3 == 1) ? frame(T_LOG, (uint8_t)i, std::vector<uint8_t>(1 + i % 240, 'x'))
                                              : frame(T_NOTE_OUT, (uint8_t)i, std::vector<uint8_t>(8, (uint8_t)i));
      s.insert(s.end(), f.begin(), f.end());
    }
    for (size_t chunk : {1u, 7u, 64u, 513u, 5000u}) {
      Parser p;
      std::vector<Got> got;
      for (size_t i = 0; i < s.size(); i += chunk) {
        p.feed(s.data() + i, i + chunk <= s.size() ? chunk : s.size() - i, collect, &got);
      }
      CHECK(got.size() == 300 && p.stats().bytes_skipped == 0 && p.pending() == 0);
      bool order = got.size() == 300;
      for (size_t i = 0; order && i < got.size(); ++i) order = got[i].seq == (uint8_t)i;
      CHECK(order);
    }
  }
}

// ------------------------------------------------------------------ engine
static uint32_t g_clock = 1000;
static uint32_t fake_micros() { return g_clock += 5; }
static std::vector<uint8_t> g_tx;
static void sink(const uint8_t* d, size_t n, void*) { g_tx.insert(g_tx.end(), d, d + n); }
static size_t info(char* buf, size_t cap, void*) { return (size_t)snprintf(buf, cap, "chip=test;psram=123"); }

static std::vector<Got> talk(Engine& e, const std::vector<uint8_t>& bytes) {
  g_tx.clear();
  e.feed(bytes.data(), bytes.size());
  Parser p;
  std::vector<Got> got;
  p.feed(g_tx.data(), g_tx.size(), collect, &got);
  return got;
}

static std::vector<uint8_t> config_payload(uint8_t plugin, uint8_t tonic, uint8_t M, uint16_t K,
                                           const std::vector<uint8_t>& pos) {
  std::vector<uint8_t> p(CONFIG_HEADER_LEN);
  p[0] = plugin;
  p[1] = tonic;
  p[2] = M;
  p[3] = 0;
  put_u16(&p[4], K);
  put_u16(&p[6], (uint16_t)pos.size());
  p.insert(p.end(), pos.begin(), pos.end());
  return p;
}

static std::vector<uint8_t> note_payload(uint16_t i, uint8_t midi, float onset, float prev_dur, bool has_prev) {
  NoteIn n{i, midi, has_prev, onset, prev_dur};
  std::vector<uint8_t> p(NOTE_IN_LEN);
  pack_note_in(n, p.data());
  return p;
}

static void test_engine() {
  static Engine e;
  e.begin(sink, fake_micros, info, nullptr, "unit", "9.9.9");

  // PING -> PONG with the same seq, the echoed host time and a device time.
  {
    auto got = talk(e, frame(T_PING, 33, {0x78, 0x56, 0x34, 0x12}));
    CHECK(got.size() == 1 && got[0].type == T_PONG && got[0].seq == 33);
    CHECK(got.size() == 1 && get_u32(got[0].payload.data()) == 0x12345678 && get_u32(got[0].payload.data() + 4) > 0);
  }
  // HELLO -> INFO.
  {
    auto got = talk(e, frame(T_HELLO, 5, {}));
    CHECK(got.size() == 1 && got[0].type == T_INFO && got[0].seq == 5);
    std::string s = got.empty() ? "" : std::string(got[0].payload.begin(), got[0].payload.end());
    CHECK(s == "fw=unit;ver=9.9.9;proto=2;plugins=echo,third_below;chip=test;psram=123");
  }
  // NOTE_IN before CONFIG: silence, status 1.
  {
    auto got = talk(e, frame(T_NOTE_IN, 1, note_payload(0, 60, 0.f, 0.f, false)));
    NoteOut o{};
    CHECK(got.size() == 1 && got[0].type == T_NOTE_OUT && parse_note_out(got[0].payload.data(), 8, &o));
    CHECK(o.midi == -1 && o.status == NOTE_NOT_CONFIGURED);
  }
  // Unknown plug-in.
  {
    auto got = talk(e, frame(T_CONFIG, 2, config_payload(200, 0, 4, 0, {})));
    CHECK(got.size() == 1 && got[0].type == T_CONFIG_ACK && got[0].seq == 2);
    CHECK(got.size() == 1 && got[0].payload[0] == ACK_UNKNOWN_PLUGIN && got[0].payload[1] == 200);
  }
  // A CONFIG whose n_pos does not match its length.
  {
    std::vector<uint8_t> p = config_payload(H_THIRD_BELOW, 0, 4, 0, {0, 1, 2, 3});
    put_u16(&p[6], 9);
    auto got = talk(e, frame(T_CONFIG, 3, p));
    CHECK(got.size() == 1 && got[0].type == T_CONFIG_ACK && got[0].payload[0] == ACK_BAD_CONFIG);
    CHECK(e.plugin() == nullptr);
  }
  // third_below in C major. Worked by hand from the Python: scale notes go two scale steps down, others
  // three semitones down, then octaves until inside F3..F5 (53..77).
  {
    auto got = talk(e, frame(T_CONFIG, 4, config_payload(H_THIRD_BELOW, 0, 4, 16, {0, 1, 2, 3, 0, 1, 2, 3})));
    CHECK(got.size() == 1 && got[0].payload[0] == ACK_OK && get_u16(got[0].payload.data() + 2) == 8);
    struct {
      int in, want;
    } cases[] = {{60, 57},   // C4 -> A3
                 {62, 59},   // D4 -> B3
                 {64, 60},   // E4 -> C4
                 {65, 62},   // F4 -> D4
                 {67, 64},   // G4 -> E4
                 {69, 65},   // A4 -> F4
                 {71, 67},   // B4 -> G4
                 {72, 69},   // C5 -> A4
                 {61, 58},   // C#4 (not in the scale) -> three semitones down
                 {84, 69},   // C6 -> A5 (81) is above F5 -> A4
                 {48, 57},   // C3 -> A2 (45) is below F3 -> A3
                 {55, 64}};  // G3 -> E3 (52) is below F3 -> E4
    uint16_t i = 0;
    for (auto c : cases) {
      auto g = talk(e, frame(T_NOTE_IN, (uint8_t)(10 + i), note_payload(i, (uint8_t)c.in, i * 1.0f, 1.0f, i > 0)));
      NoteOut o{};
      CHECK(g.size() == 1 && g[0].type == T_NOTE_OUT && g[0].seq == (uint8_t)(10 + i));
      CHECK(g.size() == 1 && parse_note_out(g[0].payload.data(), 8, &o) && o.i == i && o.status == NOTE_OK);
      CHECK(o.midi == c.want);
      if (o.midi != c.want) printf("  third_below(%d) = %d, want %d\n", c.in, o.midi, c.want);
      ++i;
    }
  }
  // The same melody in another key: tonic G (7). B4 (71) is scale degree 3 -> G4 (67); F4 (65) is not in G major.
  {
    talk(e, frame(T_CONFIG, 5, config_payload(H_THIRD_BELOW, 7, 3, 0, {})));
    auto g = talk(e, frame(T_NOTE_IN, 1, note_payload(0, 71, 0.f, 0.f, false)));
    CHECK(g.size() == 1 && (int8_t)g[0].payload[2] == 67);
    g = talk(e, frame(T_NOTE_IN, 2, note_payload(1, 65, 1.f, 1.f, true)));
    CHECK(g.size() == 1 && (int8_t)g[0].payload[2] == 62);
  }
  // echo answers her note.
  {
    talk(e, frame(T_CONFIG, 6, config_payload(H_ECHO, 0, 4, 0, {})));
    auto g = talk(e, frame(T_NOTE_IN, 3, note_payload(7, 99, 2.5f, 0.5f, true)));
    CHECK(g.size() == 1 && g[0].payload[2] == 99 && get_u16(g[0].payload.data()) == 7);
  }
  // Frames the engine does not own go to the passthrough callback.
  {
    std::vector<Got> passed;
    e.set_passthrough(collect, &passed);
    auto g = talk(e, frame(T_LED, 1, {2}));
    CHECK(g.empty() && passed.size() == 1 && passed[0].type == T_LED && passed[0].payload[0] == 2);
    e.set_passthrough(nullptr, nullptr);
  }
  // LOG is truncated to MAX_TEXT.
  {
    g_tx.clear();
    e.log(std::string(500, 'a').c_str());
    Parser p;
    std::vector<Got> got;
    p.feed(g_tx.data(), g_tx.size(), collect, &got);
    CHECK(got.size() == 1 && got[0].type == T_LOG && got[0].payload.size() == MAX_TEXT);
  }
  // A partial frame followed by silence is dropped by idle(), and the link works again.
  {
    const uint8_t half[] = {0xA5, 0x5A, T_LOG, 0, 100, 0, 'x'};
    g_tx.clear();
    e.feed(half, sizeof(half));
    g_clock += Engine::STALE_US + 10;
    for (size_t k = 0; k < sizeof(half); ++k) {
      e.idle();
      g_clock += Engine::STALE_US + 10;
    }
    auto g = talk(e, frame(T_PING, 77, {0, 0, 0, 0}));
    CHECK(g.size() == 1 && g[0].type == T_PONG && g[0].seq == 77);
  }
  CHECK(e.stats().pings == 2 && e.stats().hellos == 1 && e.stats().notes_unconfigured == 1);
}

// ------------------------------------------------------------------ vectors from Python
// File format (tools/make_note_vectors.py):
//   H <plugin name> <plugin id>
//   S <dataset> <song index> <tonic> <M> <K> <n_pos> <n_notes>
//   P <pos ...>
//   N <i> <midi> <has_prev> <onset> <prev_dur> <expected>
static int replay_vectors(const char* path) {
  FILE* fh = fopen(path, "r");
  if (!fh) {
    printf("cannot open %s\n", path);
    return 1;
  }
  char* line = nullptr;
  size_t cap = 0;
  Harmonizer* h = nullptr;
  std::vector<uint8_t> pos;
  long songs = 0, notes = 0, bad = 0;
  char name[64] = "";
  int id = -1;
  while (getline(&line, &cap, fh) > 0) {
    if (line[0] == 'H') {
      if (sscanf(line, "H %63s %d", name, &id) != 2) return 1;
      h = harmonizer((uint8_t)id);
      if (!h || strcmp(h->name(), name) != 0) {
        printf("vector file wants plug-in %s (%d), the core has %s\n", name, id, h ? h->name() : "none");
        return 1;
      }
    } else if (line[0] == 'S') {
      char ds[32];
      int idx, tonic, M, K, n_pos, n_notes;
      if (!h || sscanf(line, "S %31s %d %d %d %d %d %d", ds, &idx, &tonic, &M, &K, &n_pos, &n_notes) != 7) return 1;
      if (getline(&line, &cap, fh) <= 0 || line[0] != 'P') return 1;
      pos.clear();
      char* q = line + 1;
      for (int k = 0; k < n_pos; ++k) pos.push_back((uint8_t)strtol(q, &q, 10));
      Meta meta{tonic, M, K, pos.data(), n_pos};
      h->start(meta);
      ++songs;
    } else if (line[0] == 'N') {
      int i, midi, has_prev, want;
      float onset, prev_dur;
      if (!h || sscanf(line, "N %d %d %d %f %f %d", &i, &midi, &has_prev, &onset, &prev_dur, &want) != 6) return 1;
      int got = h->step(i, onset, midi, has_prev ? prev_dur : -1.0f);
      ++notes;
      if (got != want) {
        if (++bad <= 10) printf("  song %ld note %d: her %d -> %d, Python says %d\n", songs, i, midi, got, want);
      }
    }
  }
  free(line);
  fclose(fh);
  printf("vectors: plug-in %s, %ld songs, %ld notes, %ld differ from Python\n", name, songs, notes, bad);
  return (bad == 0 && notes > 0) ? 0 : 1;
}

int main(int argc, char** argv) {
  test_protocol();
  test_engine();
  printf("unit tests: %d checks, %d failed\n", g_checks, g_fail);
  int rc = g_fail ? 1 : 0;
  for (int k = 1; k < argc; ++k) rc |= replay_vectors(argv[k]);
  return rc;
}
