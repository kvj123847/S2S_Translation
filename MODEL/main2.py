import os
import asyncio
import threading
import tempfile
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from typing import Dict, Any
import subprocess
import time
import msvcrt

# ✅ Change to correct directory
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)
print(f"📁 Server running from: {current_directory}")

# Import your model exactly as it is
from model2 import LowLatencyTranslator, warm_up_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Low Latency Translator API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

# ✅ GLOBAL MODELS
class GlobalTranslator:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("🚀 LOADING MODELS (First time only)...")
                    cls._instance = LowLatencyTranslator()
                    warm_up_models(
                        cls._instance.asr_model,
                        cls._instance.asr_processor,
                        cls._instance.translator.model,
                        cls._instance.translator.tokenizer
                    )
                    print("✅ MODELS LOADED!")
        return cls._instance

# Pre-load models
global_translator = GlobalTranslator.get_instance()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.audio_processors: Dict[WebSocket, 'AudioProcessor'] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        translator = GlobalTranslator.get_instance()
        self.audio_processors[websocket] = AudioProcessor(websocket, translator)
        
        print(f"✅ WebSocket connected. Total: {len(self.active_connections)}")
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to translation server",
            "current_languages": {
                "source": translator.source_lang,
                "target": translator.target_lang
            }
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.audio_processors:
            self.audio_processors[websocket].stop_processing()
            del self.audio_processors[websocket]
        print(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")

    def get_processor(self, websocket: WebSocket):
        return self.audio_processors.get(websocket)

manager = ConnectionManager()

class AudioProcessor:
    def __init__(self, websocket: WebSocket, translator: LowLatencyTranslator):
        self.websocket = websocket
        self.translator = translator
        self.is_processing = False
        self.is_ott_processing = False
        self.processing_task = None
        self.ott_task = None
        self.ffmpeg_process = None

    async def change_languages(self, source_lang: str, target_lang: str):
        """Change languages and notify all connected clients"""
        try:
            # Update the translator instance
            self.translator.source_lang = source_lang
            self.translator.target_lang = target_lang
            
            print(f"🌐 LANGUAGES CHANGED: {source_lang} → {target_lang}")
            
            # Notify this client
            await self.websocket.send_json({
                "type": "languages_changed",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "message": f"Languages changed to {source_lang} → {target_lang}"
            })
            
            # Also update the global instance for new connections
            global_translator = GlobalTranslator.get_instance()
            global_translator.source_lang = source_lang
            global_translator.target_lang = target_lang
            
        except Exception as e:
            print(f"❌ Language change error: {e}")
            await self.websocket.send_json({
                "type": "error",
                "message": f"Failed to change languages: {str(e)}"
            })

    async def start_processing(self):
        """Start live translation - WITH CHUNKING"""
        if self.is_processing:
            return
        
        self.is_processing = True
        await self.websocket.send_json({"type": "session_started"})
        
        self.processing_task = asyncio.create_task(self._live_translation())

    async def _live_translation(self):
        """Live translation WITH CHUNKING"""
        try:
            import sounddevice as sd
            
            print(f"🎙 LIVE TRANSLATION STARTED: {self.translator.source_lang} → {self.translator.target_lang}")
            self.translator.is_recording = True

            while self.translator.is_recording and self.is_processing:
                # Record audio CHUNK
                recording = sd.rec(
                    int(self.translator.chunk_duration * self.translator.sample_rate),
                    samplerate=self.translator.sample_rate, 
                    channels=1, 
                    dtype='float32'
                )
                sd.wait()

                # Process audio CHUNK
                audio_data = recording.flatten()
                transcription = self.translator.speech_to_text(audio_data)
                
                if transcription and transcription.strip():
                    print(f"🎯 [{self.translator.source_lang}] ASR: {transcription}")
                    
                    # Get translation
                    translated_text = self.translator.translator.translate_text(
                        transcription, 
                        self.translator.source_lang, 
                        self.translator.target_lang
                    )
                    
                    print(f"🌐 [{self.translator.target_lang}] Translation: {translated_text}")
                    
                    # ✅ UI UPDATE
                    await self.websocket.send_json({
                        "type": "transcription_update",
                        "transcription": transcription,
                        "translated": translated_text,
                        "source_lang": self.translator.source_lang,
                        "target_lang": self.translator.target_lang
                    })
                    
                    # ✅ TTS
                    print(f"🔊 Playing {self.translator.target_lang} TTS: {translated_text}")
                    self.translator.tts.text_to_speech(translated_text, self.translator.target_lang)
                
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Live translation error: {e}")
            await self.websocket.send_json({"type": "error", "message": str(e)})
        finally:
            self.translator.is_recording = False

    async def start_ott_translation(self):
        """Start OTT translation (System Audio)"""
        if self.is_ott_processing:
            return
        
        self.is_ott_processing = True
        await self.websocket.send_json({"type": "ott_session_started"})
        
        self.ott_task = asyncio.create_task(self._ott_translation())

    async def _ott_translation(self):
        """OTT Translation using FFmpeg"""
        try:
            print(f"🎧 OTT TRANSLATION STARTED: {self.translator.source_lang} → {self.translator.target_lang}")
            
            # Use FFmpeg to capture system audio
            stereo_mix_device = "Stereo Mix (Realtek(R) Audio)"
            cmd = [
                "ffmpeg",
                "-f", "dshow",
                "-i", f"audio={stereo_mix_device}",
                "-ac", "1",
                "-ar", str(self.translator.sample_rate),
                "-f", "s16le",
                "-acodec", "pcm_s16le", 
                "pipe:1"
            ]
            
            print(f"🎯 Starting FFmpeg with: {stereo_mix_device}")
            self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            # Calculate chunk size (16-bit = 2 bytes per sample)
            bytes_per_chunk = int(self.translator.chunk_duration * self.translator.sample_rate * 2)

            while self.is_ott_processing:
                # Read audio chunk from FFmpeg pipe
                raw_audio = self.ffmpeg_process.stdout.read(bytes_per_chunk)
                if raw_audio:
                    # Convert to float32 numpy array
                    audio_np = self._pcm16le_to_float32(raw_audio)
                    
                    # Process if we have enough audio
                    if len(audio_np) > int(0.5 * self.translator.sample_rate):
                        transcription = self.translator.speech_to_text(audio_np)
                        if transcription and transcription.strip():
                            print(f"🎯 [{self.translator.source_lang}] OTT ASR: {transcription}")
                            
                            # Get translation
                            translated_text = self.translator.translator.translate_text(
                                transcription, 
                                self.translator.source_lang, 
                                self.translator.target_lang
                            )
                            
                            print(f"🌐 [{self.translator.target_lang}] OTT Translation: {translated_text}")
                            
                            # ✅ UI UPDATE
                            await self.websocket.send_json({
                                "type": "transcription_update",
                                "transcription": transcription,
                                "translated": translated_text,
                                "source_lang": self.translator.source_lang,
                                "target_lang": self.translator.target_lang,
                                "mode": "ott"
                            })
                            
                            # ✅ TTS
                            print(f"🔊 Playing {self.translator.target_lang} OTT TTS: {translated_text}")
                            self.translator.tts.text_to_speech(translated_text, self.translator.target_lang)

                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"OTT translation error: {e}")
            await self.websocket.send_json({"type": "error", "message": str(e)})
        finally:
            self._stop_ott_process()

    def _pcm16le_to_float32(self, raw_bytes):
        """Convert PCM16 Little Endian to float32 numpy array"""
        if not raw_bytes:
            return np.array([], dtype=np.float32)
        # Convert from bytes to int16, then to float32
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def _stop_ott_process(self):
        """Stop FFmpeg process"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.kill()
                self.ffmpeg_process = None
            except:
                pass

    def stop_processing(self):
        """Stop all processing"""
        self.is_processing = False
        self.is_ott_processing = False
        self.translator.is_recording = False
        
        # Stop tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.ott_task:
            self.ott_task.cancel()
        
        # Stop FFmpeg
        self._stop_ott_process()

    def stop_ott_processing(self):
        """Stop only OTT processing"""
        self.is_ott_processing = False
        if self.ott_task:
            self.ott_task.cancel()
        self._stop_ott_process()

    async def process_audio_file(self, file_path: str):
        """Process audio file - NO CHUNKING"""
        try:
            print(f"🎵 PROCESSING AUDIO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            await self.websocket.send_json({
                "type": "file_processing_started",
                "file_type": "audio",
                "source_lang": self.translator.source_lang,
                "target_lang": self.translator.target_lang
            })

            import librosa
            
            # Load entire audio file
            audio, sr = librosa.load(file_path, sr=self.translator.sample_rate)
            duration = len(audio) / sr
            print(f"✅ Loaded audio file ({duration:.1f}s)")
            
            # ✅ PROCESS ENTIRE FILE AT ONCE - NO CHUNKING
            transcription = self.translator.speech_to_text(audio)
            
            if transcription and transcription.strip():
                print(f"🎯 [{self.translator.source_lang}] File ASR: {transcription}")
                
                # Get translation
                translated = self.translator.translator.translate_text(
                    transcription, 
                    self.translator.source_lang, 
                    self.translator.target_lang
                )
                print(f"🌐 [{self.translator.target_lang}] File Translation: {translated}")
                
                # ✅ UI UPDATE
                await self.websocket.send_json({
                    "type": "transcription_update",
                    "transcription": transcription,
                    "translated": translated,
                    "source_lang": self.translator.source_lang,
                    "target_lang": self.translator.target_lang
                })
                
                # ✅ TTS
                print(f"🔊 Playing {self.translator.target_lang} TTS for file: {translated}")
                self.translator.tts.text_to_speech(translated, self.translator.target_lang)
            
            await self.websocket.send_json({
                "type": "file_processing_completed",
                "file_type": "audio"
            })
            print("✅ Audio file processing completed")

        except Exception as e:
            logger.error(f"Audio file processing error: {e}")
            await self.websocket.send_json({
                "type": "error",
                "message": f"Audio processing failed: {str(e)}"
            })

    async def process_video_file(self, file_path: str):
        """Process video file - NO CHUNKING"""
        try:
            print(f"🎥 PROCESSING VIDEO FILE: {self.translator.source_lang} → {self.translator.target_lang}")
            await self.websocket.send_json({
                "type": "file_processing_started", 
                "file_type": "video",
                "source_lang": self.translator.source_lang,
                "target_lang": self.translator.target_lang
            })

            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
                audio_path = audio_tmp.name
            
            command = [
                "ffmpeg", "-i", file_path, "-ac", "1", "-ar", "16000", 
                "-loglevel", "quiet", "-y", audio_path
            ]
            subprocess.run(command, check=True)
            
            print("✅ Video audio extracted")
            
            # Process the extracted audio as WHOLE file
            await self.process_audio_file(audio_path)
            
            # Cleanup
            os.unlink(audio_path)

        except Exception as e:
            logger.error(f"Video file processing error: {e}")
            await self.websocket.send_json({
                "type": "error",
                "message": f"Video processing failed: {str(e)}"
            })

@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open("interface.html", 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Server running - translator.html not found</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(data, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def handle_websocket_message(data: dict, websocket: WebSocket):
    message_type = data.get("type")
    processor = manager.get_processor(websocket)
    
    if not processor:
        return
    
    try:
        if message_type == "start_session":
            await processor.start_processing()
        elif message_type == "start_ott_session":
            await processor.start_ott_translation()
        elif message_type == "stop_session":
            processor.stop_processing()
            await websocket.send_json({"type": "session_stopped"})
        elif message_type == "stop_ott_session":
            processor.stop_ott_processing()
            await websocket.send_json({"type": "ott_session_stopped"})
        elif message_type == "change_languages":
            source_lang = data.get("source_lang", "english")
            target_lang = data.get("target_lang", "hindi")
            # ✅ Use the new change_languages method
            await processor.change_languages(source_lang, target_lang)
            
    except Exception as e:
        logger.error(f"Message handling error: {e}")

@app.post("/api/translate_audio")
async def translate_audio_file(file: UploadFile = File(...)):
    """Process audio file - NO CHUNKING"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Use WebSocket processing for UI updates
        if manager.active_connections:
            processor = manager.get_processor(manager.active_connections[0])
            if processor:
                print("🚀 Processing audio file")
                await processor.process_audio_file(tmp_path)
            else:
                print("❌ No processor found")
        else:
            print("❌ No active WebSocket connections")
        
        os.unlink(tmp_path)
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate_video")
async def translate_video_file(file: UploadFile = File(...)):
    """Process video file - NO CHUNKING"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Use WebSocket processing for UI updates
        if manager.active_connections:
            processor = manager.get_processor(manager.active_connections[0])
            if processor:
                print("🚀 Processing video file")
                await processor.process_video_file(tmp_path)
            else:
                print("❌ No processor found")
        else:
            print("❌ No active WebSocket connections")
        
        os.unlink(tmp_path)
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/change_languages")
async def change_languages_api(source_lang: str = Form(...), target_lang: str = Form(...)):
    """Change languages via HTTP API"""
    try:
        # Update global translator
        translator = GlobalTranslator.get_instance()
        translator.source_lang = source_lang
        translator.target_lang = target_lang
        
        # Update all connected WebSocket processors
        for websocket, processor in manager.audio_processors.items():
            processor.translator.source_lang = source_lang
            processor.translator.target_lang = target_lang
        
        print(f"🌐 LANGUAGES CHANGED GLOBALLY: {source_lang} → {target_lang}")
        
        return JSONResponse({
            "status": "success",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "message": f"Languages changed to {source_lang} → {target_lang}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    translator = GlobalTranslator.get_instance()
    return JSONResponse({
        "status": "healthy", 
        "active_connections": len(manager.active_connections),
        "current_languages": {
            "source": translator.source_lang,
            "target": translator.target_lang
        }
    })

if __name__ == "__main__":
    print("🚀 TRANSLATOR SERVER STARTING...")
    print("🌐 http://localhost:8000")
    
    uvicorn.run(
        "main2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )