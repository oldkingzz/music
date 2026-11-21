#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include <FS.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------- Display ----------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ---------- Microphone (I2S RX) ----------
#define MIC_PORT      I2S_NUM_0
#define MIC_BCK_PIN   11
#define MIC_WS_PIN    12
#define MIC_SD_PIN    13

// ---------- Speaker (I2S TX) ----------
#define SPK_PORT      I2S_NUM_1
#define SPK_BCK_PIN   4
#define SPK_WS_PIN    5
#define SPK_DOUT_PIN  6

// ---------- SD card ----------
#define SD_CS_PIN   15
#define SD_MOSI_PIN 16
#define SD_MISO_PIN 18
#define SD_SCK_PIN  17

#define BUTTON_PIN  21

// ---------- Audio ----------
#define SAMPLE_RATE      16000   // same for record + playback
#define WAVE_HEADER_SIZE 44
#define MIC_SHIFT        14      // 16 = unity gain for 24-bit mic data, 14 = ~4x. Raise if too loud.
#define CHUNK_SAMPLES    512

enum State { IDLE, RECORDING, PLAYING };
State state = IDLE;

File file;
int fileIndex = 1;
char filename[32];

// ---------------------------------------------------------------
void showMsg(const char *l1, const char *l2 = nullptr) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 20);
  display.println(l1);
  if (l2) display.println(l2);
  display.display();
}

void writeWavHeader(File &f, uint32_t sampleRate, uint16_t bitsPerSample,
                    uint16_t channels, uint32_t dataSize) {
  uint8_t h[WAVE_HEADER_SIZE];
  uint32_t fileSize = dataSize + WAVE_HEADER_SIZE - 8;
  uint32_t byteRate = sampleRate * channels * (bitsPerSample / 8);
  uint16_t blockAlign = channels * (bitsPerSample / 8);
  uint32_t fmtSize = 16;
  uint16_t pcm = 1;

  memcpy(h, "RIFF", 4);
  memcpy(h + 4, &fileSize, 4);
  memcpy(h + 8, "WAVEfmt ", 8);
  memcpy(h + 16, &fmtSize, 4);
  memcpy(h + 20, &pcm, 2);
  memcpy(h + 22, &channels, 2);
  memcpy(h + 24, &sampleRate, 4);
  memcpy(h + 28, &byteRate, 4);
  memcpy(h + 32, &blockAlign, 2);
  memcpy(h + 34, &bitsPerSample, 2);
  memcpy(h + 36, "data", 4);
  memcpy(h + 40, &dataSize, 4);

  f.write(h, WAVE_HEADER_SIZE);
}

// ---------------------------------------------------------------
void setupMic() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  cfg.use_apll = false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = MIC_BCK_PIN;
  pins.ws_io_num = MIC_WS_PIN;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = MIC_SD_PIN;

  i2s_driver_install(MIC_PORT, &cfg, 0, NULL);
  i2s_set_pin(MIC_PORT, &pins);
  i2s_zero_dma_buffer(MIC_PORT);
}

void setupSpeaker() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;            // APLL isn't available on ESP32-S3
  cfg.tx_desc_auto_clear = true;   // output silence when no data

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = SPK_BCK_PIN;
  pins.ws_io_num = SPK_WS_PIN;
  pins.data_out_num = SPK_DOUT_PIN;
  pins.data_in_num = I2S_PIN_NO_CHANGE;

  i2s_driver_install(SPK_PORT, &cfg, 0, NULL);
  i2s_set_pin(SPK_PORT, &pins);
  i2s_zero_dma_buffer(SPK_PORT);
}

// ---------------------------------------------------------------
void startRecording() {
  while (SD.exists("/rec" + String(fileIndex) + ".wav")) fileIndex++;
  snprintf(filename, sizeof(filename), "/rec%d.wav", fileIndex);

  file = SD.open(filename, FILE_WRITE);
  if (!file) {
    Serial.println("Could not open file!");
    showMsg("File open failed");
    return;
  }

  writeWavHeader(file, SAMPLE_RATE, 16, 1, 0);   // placeholder, fixed on stop
  i2s_zero_dma_buffer(MIC_PORT);
  i2s_start(MIC_PORT);

  showMsg("Recording...", filename);
  state = RECORDING;
}

void recordChunk() {
  int32_t in[CHUNK_SAMPLES];
  int16_t out[CHUNK_SAMPLES];
  size_t bytesRead = 0;

  i2s_read(MIC_PORT, in, sizeof(in), &bytesRead, portMAX_DELAY);
  int n = bytesRead / 4;

  for (int i = 0; i < n; i++) {
    int32_t s = in[i] >> MIC_SHIFT;
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    out[i] = (int16_t)s;
  }
  file.write((uint8_t *)out, n * 2);
}

void stopRecording() {
  i2s_stop(MIC_PORT);
  uint32_t dataSize = file.size() - WAVE_HEADER_SIZE;
  file.seek(0);
  writeWavHeader(file, SAMPLE_RATE, 16, 1, dataSize);
  file.close();

  Serial.printf("Saved %s, %u bytes of audio\n", filename, dataSize);
  showMsg("Saved:", filename);
  delay(500);
}

// ---------------------------------------------------------------
void playFile(const char *path) {
  File f = SD.open(path, FILE_READ);
  if (!f) {
    Serial.println("Playback: could not open file");
    showMsg("Playback failed");
    return;
  }
  f.seek(WAVE_HEADER_SIZE);   // skip WAV header

  showMsg("Playing...", path);
  state = PLAYING;
  i2s_zero_dma_buffer(SPK_PORT);

  uint8_t buf[1024];
  while (f.available()) {
    // press button to abort playback
    if (digitalRead(BUTTON_PIN) == LOW) {
      delay(200);
      while (digitalRead(BUTTON_PIN) == LOW);
      break;
    }

    int n = f.read(buf, sizeof(buf));
    int offset = 0;
    while (n > 0) {
      size_t written = 0;
      i2s_write(SPK_PORT, buf + offset, n, &written, portMAX_DELAY);
      offset += written;
      n -= written;
    }
  }
  f.close();

  // flush: let the DMA buffers drain, then silence
  delay(100);
  i2s_zero_dma_buffer(SPK_PORT);

  showMsg("Done. Press to", "record again");
  state = IDLE;
}

// ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
    for (;;);
  }
  showMsg("Display Setup");

  SPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  if (!SD.begin(SD_CS_PIN)) {
    showMsg("SD Card not", "Initialized!");
    while (1) delay(100);
  }

  setupMic();
  i2s_stop(MIC_PORT);   // idle until recording starts
  setupSpeaker();

  showMsg("System ready", "Press to record");
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(200);   // debounce
    if (state == IDLE) {
      startRecording();
    } else if (state == RECORDING) {
      stopRecording();
      playFile(filename);   // auto playback
    }
    while (digitalRead(BUTTON_PIN) == LOW);
  }

  if (state == RECORDING) {
    recordChunk();
  }
}