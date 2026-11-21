#include "audio_io.h"

#include <string.h>

#include "config.h"
#include "esp_attr.h"

namespace duet {

static bool IRAM_ATTR on_rx_overflow(i2s_chan_handle_t, i2s_event_data_t*, void* ctx) {
  volatile uint32_t* c = (volatile uint32_t*)ctx;
  if (c) *c = *c + 1;
  return false;
}

int audio_begin(AudioIO& io, volatile uint32_t* dma_ovf_counter) {
  esp_err_t e;

  // ---------------- I2S0 RX: PCM1808 in slave mode (MD1=MD0=0), I2S format (FMT=0).
  // PCM1808 slave mode requires BCK = 64 fs (datasheet SLES177B, "Audio data interface
  // timing (slave mode)": t(BCKP) = 1/(64 fS)) and SCKI = 256/384/512 fs synchronous with
  // LRCK. 32-bit slots x 2 = 64 BCK per frame; MCLK = 256 fs = 6.144 MHz -> BCLK = MCLK/4.
  i2s_chan_config_t rx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  rx_cc.dma_desc_num = I2S_RX_DMA_DESC;
  rx_cc.dma_frame_num = I2S_RX_DMA_FRAMES;
  e = i2s_new_channel(&rx_cc, nullptr, &io.rx);
  if (e != ESP_OK) return e;

  i2s_std_config_t rx_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_FS),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
      .gpio_cfg =
          {
              .mclk = (gpio_num_t)PIN_ADC_MCLK,
              .bclk = (gpio_num_t)PIN_ADC_BCK,
              .ws = (gpio_num_t)PIN_ADC_LRCK,
              .dout = I2S_GPIO_UNUSED,
              .din = (gpio_num_t)PIN_ADC_DIN,
              .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
          },
  };
  rx_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
  rx_cfg.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT;
  e = i2s_channel_init_std_mode(io.rx, &rx_cfg);
  if (e != ESP_OK) return e;

  i2s_event_callbacks_t cbs;
  memset(&cbs, 0, sizeof(cbs));
  cbs.on_recv_q_ovf = on_rx_overflow;
  e = i2s_channel_register_event_callback(io.rx, &cbs, (void*)dma_ovf_counter);
  if (e != ESP_OK) return e;

  // ---------------- I2S1 TX: MAX98357A, 48 kHz, 16-bit stereo (BCLK = 32 fs = 1.536 MHz,
  // allowed: "BCLK must be 32, 48, or 64X of LRCLK"). No MCLK pin needed by the amp.
  i2s_chan_config_t tx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
  tx_cc.dma_desc_num = I2S_TX_DMA_DESC;
  tx_cc.dma_frame_num = I2S_TX_DMA_FRAMES;
  tx_cc.auto_clear = true;  // DMA underrun -> zeros, never a repeated buffer (buzz)
  e = i2s_new_channel(&tx_cc, &io.tx, nullptr);
  if (e != ESP_OK) return e;

  i2s_std_config_t tx_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SPK_FS),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
      .gpio_cfg =
          {
              .mclk = I2S_GPIO_UNUSED,
              .bclk = (gpio_num_t)PIN_AMP_BCLK,
              .ws = (gpio_num_t)PIN_AMP_LRC,
              .dout = (gpio_num_t)PIN_AMP_DOUT,
              .din = I2S_GPIO_UNUSED,
              .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
          },
  };
  e = i2s_channel_init_std_mode(io.tx, &tx_cfg);
  if (e != ESP_OK) return e;

  // Preload silence so the first real frame goes straight behind it.
  static int16_t zeros[I2S_TX_DMA_FRAMES * 2];
  memset(zeros, 0, sizeof(zeros));
  for (int i = 0; i < I2S_TX_DMA_DESC; ++i) {
    size_t loaded = 0;
    i2s_channel_preload_data(io.tx, zeros, sizeof(zeros), &loaded);
    if (loaded < sizeof(zeros)) break;
  }

  // Enable TX first: the MAX98357A must never see BCLK without LRCLK, and the std driver
  // starts both together. RX second (PCM1808 needs ~8960/fs = 373 ms after clocks start
  // before DOUT is valid; those first frames are zeros from the ADC).
  e = i2s_channel_enable(io.tx);
  if (e != ESP_OK) return e;
  e = i2s_channel_enable(io.rx);
  if (e != ESP_OK) return e;

  i2s_chan_info_t info;
  if (i2s_channel_get_info(io.rx, &info) == ESP_OK) {
    io.rx_mclk_hz = info.mclk_hz;
    io.rx_bclk_hz = info.bclk_hz;
  }
  if (i2s_channel_get_info(io.tx, &info) == ESP_OK) {
    io.tx_mclk_hz = info.mclk_hz;
    io.tx_bclk_hz = info.bclk_hz;
  }
  return ESP_OK;
}

bool audio_read_mic(AudioIO& io, int16_t* mono, uint32_t timeout_ms) {
  static int32_t raw[FRAME_SAMPLES * 2];
  size_t got = 0;
  if (i2s_channel_read(io.rx, raw, sizeof(raw), &got, timeout_ms) != ESP_OK || got != sizeof(raw)) return false;
  // PCM1808: 24-bit two's complement, MSB-first, left-aligned in the 32-bit slot.
  // Keep the top 16 bits with rounding and saturation.
  for (int i = 0; i < FRAME_SAMPLES; ++i) {
    int32_t s = raw[2 * i + MIC_SLOT];
    int32_t r = (s >> 16) + ((s >> 15) & 1);
    if (r > 32767) r = 32767;
    mono[i] = (int16_t)r;
  }
  return true;
}

bool audio_write_spk(AudioIO& io, const int16_t* lr, size_t frames, uint32_t timeout_ms) {
  size_t bytes = frames * 2 * sizeof(int16_t), done = 0;
  return i2s_channel_write(io.tx, lr, bytes, &done, timeout_ms) == ESP_OK && done == bytes;
}

}  // namespace duet
