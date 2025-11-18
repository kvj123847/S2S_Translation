# Real-Time Speech-to-Speech (S2S) Translation System

This repository contains a real-time multilingual Speech-to-Speech (S2S) translation system developed as an internship project for Infosys Springboard. It supports live microphone translation, system audio (OTT) translation, and audio/video file translation with low latency and a simple web interface.

## Project Overview

The system performs end-to-end speech translation using a five-stage pipeline:

- Audio capture (microphone, system audio, or uploaded file)
- ASR (Automatic Speech Recognition) — fine-tuned Whisper
- Machine translation — NLLB-200 (distilled)
- Text-to-Speech (TTS) — gTTS (Google Text-to-Speech)
- UI rendering and audio output


## Repository Structure

- `interface.html` — Main web interface for live, OTT, and file translation
- `translator.html` — Alternate/auxiliary interface layout
- `main2.py` — FastAPI backend, WebSocket handling, session management
- `model2.py` — ASR, translation, and TTS pipeline implementation
- `requirements.txt` — Python dependencies
- `fine_tuned_whisper/` — Fine-tuned Whisper model files

## Key Components

- Fine-tuned ASR: A custom Whisper model fine-tuned.
	- Model link: https://huggingface.co/kklwq/whisper-small-hi-finetuned
- Translation: NLLB-200 (600M distilled) for high-quality multilingual translation.
- TTS: gTTS for speech synthesis (network-dependent).

## Technology Stack

- Backend: FastAPI, Uvicorn, WebSockets
- Audio/tools: FFmpeg, sounddevice, librosa, pygame
- Models: Whisper (fine-tuned), NLLB-200, gTTS
- Frontend: HTML5, CSS3, JavaScript, Bootstrap 5, Font Awesome

## System Architecture (high level)

- `GlobalTranslator` loads and caches models (singleton) to avoid repeated initialization.
- `AudioProcessor` manages each user session and audio pipelines.
- Asynchronous WebSockets provide low-latency, real-time communication.
- TTS runs in background threads to avoid blocking the main event loop.
- Translation caching and warm-up reduce repeated inference and cold-start latency.

## Features

- Live microphone translation (real-time transcription, translation, playback)
- OTT / system audio translation (captures system audio via FFmpeg / Stereo Mix)
- File-based translation: supports `.wav`, `.mp3`, `.mp4`, `.avi`
- Language selection for source and target languages
- Low-latency optimizations (warm-up, async design, caching)

## Installation & Setup

Requirements:

- Windows 10/11 recommended (project tested on Windows)
- Python 3.9+
- FFmpeg installed and added to `PATH`
- (Optional) NVIDIA GPU for faster model inference

Quick start (Windows PowerShell):

```powershell
git clone <repo-url>
cd <repo-folder>
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main2.py
# Open http://localhost:8000 in your browser
```

Note: For OTT/system audio capture, enable `Stereo Mix` in Windows Sound Settings (Recording tab) if required by your system.

## Usage

- Live Translation: open the web UI and start live translation to stream from the microphone.
- OTT Translation: enable system audio capture, play system audio, then start OTT mode in the UI.
- File Translation: upload an audio/video file in the UI and request translation.

## Known Limitations

- Stereo Mix device name may be hard-coded and require manual adjustment.
- gTTS depends on network calls and may introduce latency.
- Speaker diarization (multi-speaker separation) is not implemented.
- Cloud deployment and cross-platform audio capture (Linux/macOS) are not provided out-of-the-box.

## Future Enhancements

- Replace gTTS with a local TTS (e.g., FastSpeech2, Bark) for lower latency and offline use
- Add speaker diarization and multi-speaker support
- Add real-time subtitle overlays for OTT mode
- Provide Docker images and CI/CD for cloud deployment
- Add Linux/macOS audio support (PulseAudio, BlackHole)

## Recommended `.gitignore`

- `__pycache__/`
- `*.py[cod]`
- `*.mp3`
- `*.wav`
- `*.tmp`
- `venv/`
- `.env`



