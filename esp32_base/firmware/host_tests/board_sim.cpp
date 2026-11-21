// Mac stand-in for the board. It is the same DuetCore the sketch compiles, with stdin as "bytes from the
// host" and stdout as "bytes to the host". host/duet_host/link.py starts it as a child process, so every
// check that will run against the real board tomorrow can run tonight against exactly the same C++.
// What it cannot show: USB behaviour, timing on the real chip, PSRAM, anything electrical.
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>

#include "DuetCore.h"

static uint32_t now_us() {
  static const auto t0 = std::chrono::steady_clock::now();
  return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
      .count();
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

static size_t info(char* buf, size_t cap, void*) {
  return (size_t)snprintf(buf, cap, "chip=sim;flash=0;psram=0;sim=1");
}

int main() {
  static duet::Engine engine;
  engine.begin(write_all, now_us, info, nullptr, "board_sim", "0.1.0");
  static uint8_t buf[4096];
  struct pollfd pfd = {0, POLLIN, 0};
  for (;;) {
    int r = poll(&pfd, 1, 20);
    if (r < 0 && errno == EINTR) continue;
    if (r == 0) {
      engine.idle();
      continue;
    }
    ssize_t n = ::read(0, buf, sizeof(buf));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return 0;  // stdin closed: the host is done
    engine.feed(buf, (size_t)n);
  }
}
