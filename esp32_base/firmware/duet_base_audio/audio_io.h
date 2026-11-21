// I2S bring-up for the PCM1808 (I2S0 RX, ESP32 master, MCLK out) and the MAX98357A
// (I2S1 TX, 48 kHz). ESP-IDF 5.x "i2s_std" driver as shipped in arduino-esp32 3.x.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "driver/i2s_std.h"

namespace duet {

struct AudioIO {
  i2s_chan_handle_t rx = nullptr;  // I2S0, PCM1808
  i2s_chan_handle_t tx = nullptr;  // I2S1, MAX98357A
  uint32_t rx_mclk_hz = 0, rx_bclk_hz = 0, tx_mclk_hz = 0, tx_bclk_hz = 0;
};

// Must be called from the task that will service audio, so the I2S DMA interrupts are
// allocated on that core. dma_ovf_counter is incremented from the RX ISR on queue overflow.
// Returns ESP_OK or the first failing esp_err_t.
int audio_begin(AudioIO& io, volatile uint32_t* dma_ovf_counter);

// Blocking read of one 10 ms mic frame (240 stereo 32-bit frames) -> 240 int16 mono.
// Returns true on success.
bool audio_read_mic(AudioIO& io, int16_t* mono240, uint32_t timeout_ms);

// Blocking write of interleaved stereo int16 at 48 kHz.
bool audio_write_spk(AudioIO& io, const int16_t* lr, size_t frames48k, uint32_t timeout_ms);

}  // namespace duet
