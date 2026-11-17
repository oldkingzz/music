# ESP32 Audio-to-MIDI Piano Player

A self-contained PlatformIO/Arduino project for **M5Stack STAMP-S3A**. The ESP32 reads a monophonic WAV recording from an SPI MicroSD card, detects its melody locally, generates MIDI notes and a Standard MIDI File, and serves an offline piano player to an iPhone.

**Pitch detection runs on the ESP32, not on the computer or in JavaScript.** The browser only synthesizes and plays the detected notes. No external PSRAM, internet service, CDN, microphone driver or Web MIDI API is required.

This English-language sharing snapshot was prepared on **2026-11-17** from the working project. It preserves the firmware behavior; comments, serial diagnostics and developer documentation are in English. The original development directory and the firmware already on the board were not replaced.

## Current features

- SPI SD initialization, card diagnostics and a RIFF chunk parser that handles `LIST`/`JUNK` metadata.
- Complete WAV read tests and separate WAV/PCM SHA-256 verification.
- Standalone Wi-Fi AP, mobile webpage, original WAV playback and download.
- HTTP `HEAD`, single-byte-range requests, `Content-Range`, `Content-Length` and `If-Range` handling.
- Bounded SD prefetch and a replaceable audio-input interface; optional 10 ms realtime simulation.
- ESP32-local YIN pitch detection, confidence/silence handling, smoothing and note segmentation.
- Session/event IDs, retained MIDI events, SSE delivery and HTTP polling fallback.
- Automatic analysis after boot and a simple **Play Piano / Pause** button, piano roll and highlighted keyboard.
- Valid format-0 `/test.mid` output on the SD card and a MIDI download endpoint.
- On-device sine-wave, HTTP, SSE and integrity checks, plus independent Python validators.

Not implemented: physical I2S capture, polyphonic transcription, raw-audio UDP transport, packet-loss experiments, FEC or automatic retransmission. See [project status](docs/PROJECT_STATUS.md) for measured results and limits.

## Hardware and fixed wiring

| Part | Configuration |
|---|---|
| Controller | M5Stack STAMP ESP32-S3A / ESP32-S3FN8 |
| Flash | 8 MB |
| External PSRAM | Not required or enabled |
| SD interface | SPI, configured at 1 MHz |
| Serial monitor | Native USB Serial/JTAG, 115200 baud, UTF-8 |

| MicroSD module | STAMP-S3A |
|---|---|
| VCC | 3V3 |
| GND | GND |
| CS | GPIO9 |
| MOSI | GPIO7 |
| CLK / SCK | GPIO5 |
| MISO | GPIO3 |

The firmware explicitly calls:

```cpp
SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
```

Keep this wiring for the current board. GPIO3 has a reset-time strapping role; the documented wiring has already worked on the tested hardware. Board migration settings belong in `include/board_config.h` and `platformio.ini`.

The PlatformIO board is `m5stack-stamps3`, selected for the S3A's ESP32-S3FN8 and 8 MB flash. The platform is pinned to `espressif32@6.10.0`, using the Arduino-ESP32 2.0.17 series. The board definition supplies `ARDUINO_USB_MODE=1`; the project enables `ARDUINO_USB_CDC_ON_BOOT=1`. Do not copy unrelated ESP32-S3 PSRAM or USB settings into this project.

## Quick start

1. Install VS Code and the **PlatformIO IDE** extension.
2. Open **this folder**, containing `platformio.ini`, as the project root.
3. Power off the board and prepare a FAT32 MicroSD card.
4. Copy the included **`sd_card/test.wav`** to the card's root as **`/test.wav`**. Do not copy it into an `sd_card` directory on the card. The WAV is not embedded in the firmware and is not uploaded over USB.
5. Connect the module with the wiring above and connect the board using a USB data cable.
6. In the PlatformIO terminal, run:

   ```powershell
   pio run -e stamp_s3a_audio
   pio device list
   pio run -e stamp_s3a_audio -t upload --upload-port COM5
   pio device monitor --port COM5 --baud 115200
   ```

   Replace `COM5` with the actual port. USB bootloader and application ports may differ. Close other serial monitors before uploading.

7. Allow roughly **35-45 seconds** after a cold boot for SD checks, hashing and automatic local MIDI analysis. SD performance affects startup time.
8. On the iPhone, join:

   ```text
   SSID:     ESP32_Audio
   Password: audio12345
   URL:      http://192.168.4.1/
   ```

9. Keep the hotspot connection even if iOS reports no internet. Use **HTTP**, not HTTPS. Wait for **Piano ready. Tap Play Piano.**, then tap **Play Piano**. The same button pauses and resumes playback.

The computer can remain on its existing internet Wi-Fi. None of the included scripts changes Windows network settings. The default AP credentials are demo values; change `include/network_config.h` if your team needs different ones.

If automatic download mode fails, disconnect USB, hold **G0**, reconnect USB and release G0 before uploading. Release the button and power-cycle afterward if the application remains in download mode. G0 is not a reset button.

## Audio playback: what is actually playing?

```text
SD test.wav -> ESP32 WAV parser -> bounded input buffer
            -> ESP32 YIN -> note tracking -> MIDI events + SD test.mid
            -> Wi-Fi SSE / HTTP -> browser piano synthesis -> local audio playback
```

- **Original Audio** plays the original singing WAV.
- **Play Piano** uses the actual ESP32-generated MIDI events. The iPhone synthesizes them with `OfflineAudioContext`, encodes a temporary PCM WAV in browser memory, then plays it through an HTML audio element.
- **Download MIDI** downloads an actual Standard MIDI File containing notes and timing. Safari does not directly play the `.mid` file.

The browser-rendered WAV is an output format, not computer-side pitch detection. It is generated automatically from MIDI and is not stored on the ESP32. This playback path resolved the tested iPhone's earlier issue where keys highlighted but a realtime AudioContext was inaudible. A user tap is still required to start audible playback.

## Repository layout

```text
platformio.ini          Board, framework, USB and embedded web assets
include/               Public module interfaces and configuration
src/                   Firmware implementation
web/                   English HTML/CSS/JS embedded in flash
sd_card/test.wav       Demo recording to copy to the MicroSD root
examples/reference/    Previously measured ESP32 output, for comparison only
tools/                 Optional serial capture and validation utilities
docs/                  Architecture, API, testing and progress notes
logs/                  Local test output; generated files are ignored
artifacts/             Local generated outputs; ignored by Git
```

The reference MIDI and event JSON are **not loaded by the firmware**. The ESP32 regenerates events from the SD WAV after boot. Only the explicitly marked local UI preview tool reads reference JSON.

## Development and debugging

All firmware libraries (`FS`, `SD`, `SPI`, `WiFi`, `WebServer`, FreeRTOS and mbedTLS) come with the pinned ESP32 framework. No additional PlatformIO library installation is required.

Optional tools need Python 3.10 or newer. Install serial capture support with:

```powershell
python -m pip install -r requirements-tools.txt
```

Example serial capture after boot:

```powershell
python tools/capture_serial.py --port COM5 --command e --seconds 8 --output logs/status.log
```

Serial commands and HTTP endpoints are documented in [API reference](docs/API.md). See [testing](docs/TESTING.md) for sine-wave checks, original WAV integrity, PCM timing, MIDI validation and local UI preview.

The staged build environments remain available:

| Environment | Purpose |
|---|---|
| `stamp_s3a_sd` | SD/WAV diagnostics only |
| `stamp_s3a_wifi` | SD diagnostics plus AP |
| `stamp_s3a_web` | Web server without audio streaming enabled |
| `stamp_s3a_audio` | Full automatic Audio-to-MIDI demonstration; default |

Whenever you change `web/`, rebuild and upload the firmware. HTML, CSS and JavaScript are embedded with `board_build.embed_txtfiles`; no separate filesystem upload is needed.

## Sharing this folder on GitHub

Upload **the contents of this folder**, with `platformio.ini` and `README.md` at the repository root. Do not upload the parent development workspace. This snapshot excludes local toolchains, build caches, device logs, browser traces and network-switching helpers.

If you use Git, create an empty repository on GitHub and run the following from this folder, replacing the URL:

```text
git init
git add .
git commit -m "Share ESP32 Audio-to-MIDI piano demo"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Alternatively, use GitHub's file upload page. Hidden files such as `.gitignore` and `.gitattributes` should be included. No repository was created or published automatically. A distribution license has not been selected for this snapshot.
