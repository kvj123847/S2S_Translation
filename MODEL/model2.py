import torch
import numpy as np
import sounddevice as sd
import asyncio
import threading
import pygame
from transformers import WhisperForConditionalGeneration, WhisperProcessor, NllbTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import tempfile
import os
import librosa
from collections import OrderedDict
import subprocess
import msvcrt

def warm_up_models(asr_model, processor, translator, tokenizer):
    print("🚀 Warming up models... please wait (first-time latency only)")

    dummy_audio = np.zeros(16000)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        _ = asr_model.generate(**inputs)

    dummy_text = "Hello"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    with torch.no_grad():
        _ = translator.generate(**inputs)

    try:
        tts = gTTS(text="Test", lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            temp_path = tmp_file.name
        tts.save(temp_path)
        os.unlink(temp_path)
    except Exception as e:
        print(f"⚠️ gTTS warm-up skipped: {e}")

    print("✅ Warm-up complete. Models are ready.\n")

class GoogleTTSWrapper:
    def __init__(self):
        pygame.mixer.init()
        self.language_map = {
            'hindi': 'hi', 'english': 'en', 'bengali': 'bn', 'tamil': 'ta',
            'telugu': 'te', 'marathi': 'mr', 'gujarati': 'gu', 'kannada': 'kn',
            'malayalam': 'ml', 'punjabi': 'pa', 'odia': 'or', 'assamese': 'as', 'urdu': 'ur'
        }

    def get_language_code(self, language_name):
        return self.language_map.get(language_name.lower(), 'hi')

    def text_to_speech(self, text, language_name):
        try:
            lang_code = self.get_language_code(language_name)
            print(f"🔊 Generating {language_name.title()} TTS...")

            tts = gTTS(text=text, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name
            tts.save(temp_path)

            threading.Thread(target=self._play_audio, args=(temp_path,), daemon=True).start()
            return True

        except Exception as e:
            print(f"❌ Google TTS error: {e}")
            return False

    def _play_audio(self, audio_file):
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            pygame.mixer.music.unload()  # <-- unload first
            pygame.time.wait(150)        # <-- wait for file release
            os.unlink(audio_file)        # <-- safe delete
        except Exception as e:
            print(f"❌ Audio playback error: {e}")



class NLLBTranslator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.translation_cache = OrderedDict()
        self.cache_size = 500
        self.tokenizer = None
        self.model = None

    def load_models(self):
        print("🔄 Loading NLLB translation model...")
        try:
            self.tokenizer = NllbTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            print("✅ NLLB translation model loaded")
        except Exception as e:
            print(f"❌ NLLB loading failed: {e}")
            raise

    def translate_text(self, text, source_lang, target_lang):
        try:
            if not text.strip():
                return text

            cache_key = f"{source_lang}{target_lang}{text.strip().lower()}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]

            lang_map = {
                'english': 'eng_Latn', 'hindi': 'hin_Deva', 'bengali': 'ben_Beng',
                'tamil': 'tam_Taml', 'telugu': 'tel_Telu', 'marathi': 'mar_Deva',
                'gujarati': 'guj_Gujr', 'kannada': 'kan_Knda', 'malayalam': 'mal_Mlym',
                'punjabi': 'pan_Guru', 'odia': 'ory_Orya', 'assamese': 'asm_Beng', 'urdu': 'urd_Arab'
            }

            src_code = lang_map.get(source_lang.lower(), 'eng_Latn')
            tgt_code = lang_map.get(target_lang.lower(), 'hin_Deva')

            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )

            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            self.translation_cache[cache_key] = translated_text
            if len(self.translation_cache) > self.cache_size:
                self.translation_cache.popitem(last=False)

            return translated_text

        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text


class LowLatencyTranslator:
    def __init__(self):
        self.asr_model = None
        self.asr_processor = None
        self.translator = None
        self.tts = None

        self.sample_rate = 16000
        self.chunk_duration = 2.5
        self.source_lang = 'english'
        self.target_lang = 'hindi'
        self.is_recording = False

        self.load_models()

    def load_models(self):
        print("🔄 Loading models...")
        try:
            # ✅ Path to your trained Whisper model
            asr_path = r"C:\Users\Asus\Desktop\GGGG\Model\final_model"
            self.asr_processor = WhisperProcessor.from_pretrained(asr_path)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(asr_path)
            self.asr_model.eval()
            print("✅ ASR model loaded")

            self.translator = NLLBTranslator()
            self.translator.load_models()

            self.tts = GoogleTTSWrapper()
            print("✅ TTS initialized")

        except Exception as e:
            print(f"❌ Error loading models: {e}")

    async def start_ott_translation(self):
        """OTT Translation using FFmpeg (no feedback loop)"""
        print(f"\n🎧 STARTING OTT TRANSLATION: {self.source_lang.title()} → {self.target_lang.title()}")
        print("💡 Play YouTube/Netflix - system audio will be captured")
        print("🔊 You'll hear original audio AND translations")
        print("💡 Press Enter to stop\n")

        self.is_recording = True

        try:
            # USE FFMPEG INSTEAD OF SOUNDDEVICE
            stereo_mix_device = "Stereo Mix (Realtek(R) Audio)"
            cmd = [
                "ffmpeg",
                "-f", "dshow",
                "-i", f"audio={stereo_mix_device}",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-f", "s16le",
                "-acodec", "pcm_s16le", 
                "pipe:1"
            ]
            
            print(f"🎯 Starting FFmpeg with: {stereo_mix_device}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            # Calculate chunk size (16-bit = 2 bytes per sample)
            bytes_per_chunk = int(self.chunk_duration * self.sample_rate * 2)

            while self.is_recording:
                # Stop if Enter key is pressed
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':
                        print("\n🛑 OTT translation stopped. Returning to menu...")
                        break

                # Read audio chunk from FFmpeg pipe
                raw_audio = proc.stdout.read(bytes_per_chunk)
                if raw_audio:
                    # Convert to float32 numpy array
                    audio_np = self._pcm16le_to_float32(raw_audio)
                    
                    # Process if we have enough audio
                    if len(audio_np) > int(0.5 * self.sample_rate):
                        transcription = self.speech_to_text(audio_np)
                        if transcription:
                            self.process_translation(transcription)

                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"❌ OTT session error: {e}")
        finally:
            self.is_recording = False
            try:
                proc.kill()
            except:
                pass

    # ADD THIS NEW HELPER METHOD (put it after start_ott_translation):
    def _pcm16le_to_float32(self, raw_bytes):
        """Convert PCM16 Little Endian to float32 numpy array"""
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        # Convert from bytes to int16, then to float32
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def process_video_file(self, video_file_path):
        
        try:
            if not os.path.exists(video_file_path):
                print("❌ Video file not found")
                return

            print(f"🎥 Extracting audio from: {video_file_path}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name

            command = [
                "ffmpeg",
                "-i", video_file_path,
                "-ac", "1",
                "-ar", "16000",
                "-loglevel", "quiet",
                "-y",
                audio_path
            ]
            subprocess.run(command, check=True)

            print("✅ Audio extracted successfully")

            self.process_audio_file(audio_path)

            try:
                os.remove(audio_path)
            except:
                pass

        except subprocess.CalledProcessError:
            print("❌ FFmpeg failed to extract audio — make sure FFmpeg is in PATH")
        except Exception as e:
            print(f"❌ Video processing error: {e}")

    def speech_to_text(self, audio_data):
        try:
            inputs = self.asr_processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                generated_ids = self.asr_model.generate(
                    inputs.input_features,
                    language="en",
                    task="transcribe",
                    max_length=100
                )
            transcription = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            print(f"❌ ASR Error: {e}")
            return ""

    def process_translation(self, transcription):
        if not transcription:
            return
        print(f"🎯 Detected: {transcription}")
        translated_text = self.translator.translate_text(transcription, self.source_lang, self.target_lang)
        print(f"🌐 Translated: {translated_text}")
        self.tts.text_to_speech(translated_text, self.target_lang)

    def change_languages(self):
        print("\n🔄 CHANGING LANGUAGES")
        langs = list(self.tts.language_map.keys())
        for i, lang in enumerate(langs, 1):
            print(f"{i:2d}. {lang.title()}")
        try:
            s = int(input("Select SOURCE language: ")) - 1
            t = int(input("Select TARGET language: ")) - 1
            if 0 <= s < len(langs):
                self.source_lang = langs[s]
            if 0 <= t < len(langs):
                self.target_lang = langs[t]
            print(f"✅ {self.source_lang.title()} → {self.target_lang.title()}")
        except:
            print("❌ Invalid input")

    async def start_translation_session(self):
        print(f"\n🎙 STARTING SESSION: {self.source_lang.title()} → {self.target_lang.title()}")
        print("💡 Speak clearly. Press Enter to stop live translation.\n")

        self.is_recording = True

        try:
            while self.is_recording:
                # ✅ Stop if Enter key is pressed
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # Enter key
                        print("\n🛑 Live translation stopped. Returning to menu...")
                        break

                # 🎧 Record small audio chunk
                recording = sd.rec(int(self.chunk_duration * self.sample_rate),
                                samplerate=self.sample_rate, channels=1, dtype='float32')
                sd.wait()

                # 🧠 ASR + Translation + TTS
                transcription = self.speech_to_text(recording.flatten())
                if transcription:
                    self.process_translation(transcription)

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
        except Exception as e:
            print(f"❌ Session error: {e}")
        finally:
            self.is_recording = False


    def process_audio_file(self, path):
        try:
            if not os.path.exists(path):
                print("❌ File not found.")
                return
            audio, sr = librosa.load(path, sr=self.sample_rate)
            print(f"✅ Loaded file ({len(audio)/sr:.1f}s)")
            chunks = int(self.chunk_duration * sr)
            full_text = []
            for i in range(0, len(audio), chunks):
                chunk = audio[i:i + chunks]
                if len(chunk) < 0.5 * sr:
                    continue
                text = self.speech_to_text(chunk)
                if text:
                    full_text.append(text)
            combined = " ".join(full_text)
            print(f"🎯 Transcription: {combined}")
            translated = self.translator.translate_text(combined, self.source_lang, self.target_lang)
            print(f"🌐 Translation: {translated}")
            self.tts.text_to_speech(translated, self.target_lang)
        except Exception as e:
            print(f"❌ File processing error: {e}")

    def run(self):
        print("🚀 LOW LATENCY TRANSLATOR - READY!")
        while True:
            print("\n" + "=" * 50)
            print(f"🎙  LOW LATENCY TRANSLATOR")
            print("=" * 50)
            print(f"🌐 {self.source_lang.title()} → {self.target_lang.title()}")
            print(f"⏱ Chunk: {self.chunk_duration}s")
            print("=" * 50)
            print("1. 🎤 Start Live Translation (Microphone)")
            print("2. 📺 Start OTT Translation (System Audio)")
            print("3. 📁 Translate Audio File")
            print("4. 🎬 Translate Video File")
            print("5. 🔄 Change Languages")
            print("6. 🚪 Exit")
            print("=" * 50)
            ch = input("Select option (1-6): ").strip()
            if ch == "1":
                asyncio.run(self.start_translation_session())
            elif ch == "2":
                asyncio.run(self.start_ott_translation())
            elif ch == "3":
                path = input("Enter audio file path: ").strip().strip('"')
                self.process_audio_file(path)
            elif ch == "4":
                video_path = input("Enter video file path: ").strip().strip('"')
                if video_path:
                    self.process_video_file(video_path)
            elif ch == "5":
                self.change_languages()
            elif ch == "6":
                print("👋 Exiting...")
                break
            else:
                print("❌ Invalid choice.")


if __name__ == "__main__":
    translator = LowLatencyTranslator()
    # 🔥 Warm up models before first use
    warm_up_models(
        translator.asr_model,
        translator.asr_processor,
        translator.translator.model,
        translator.translator.tokenizer
    )
    translator.run()