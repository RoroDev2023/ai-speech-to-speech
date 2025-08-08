# realtime_bridge.py
import asyncio
import json
import logging
from enum import Enum
from typing import Any, Optional
from datetime import datetime
import base64
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient
from log import send_log, workspace_id, shared_key, log_type
import wave
import os
import uuid
import binascii

import aiohttp
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

print("✅ RTMT MODULE LOADED")

# Setup local log file
logger = logging.getLogger("voicerag")
logging.basicConfig(
    filename="chat_history.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Azure credentials (as provided in your snippet)


blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER)

# Use absolute base dir and local audio dir so file locations are deterministic
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOCAL_AUDIO_DIR = os.path.join(BASE_DIR, "recorded_audios")
os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)
print(f"✅ Local audio dir: {LOCAL_AUDIO_DIR} (cwd={os.getcwd()})")


def synthesize_and_upload(text: str, filename: str) -> str:
    """
    Synthesize text to a local file (filename must be full path) and upload to blob storage.
    Returns blob url or empty string on failure.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except Exception:
        pass

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.AudioConfig(filename=filename)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        abs_path = os.path.abspath(filename)
        print(f"✅ Saved model audio locally: {abs_path}")
        try:
            with open(filename, "rb") as data:
                blob_client = container_client.get_blob_client(blob=os.path.basename(filename))
                blob_client.upload_blob(data, overwrite=True)
                url = blob_client.url
                print(f"☁️ Uploaded model audio to Azure: {url}")
                return url
        except Exception as e:
            print(f"❌ Failed uploading model audio to blob (local saved OK): {e}")
            return ""
    else:
        print("❌ TTS failed:", result.reason)
        return ""

def save_pcm_to_wav(pcm_bytes, filename):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 2 bytes = 16-bit samples
        wf.setframerate(48000)    # 48 kHz
        wf.writeframes(pcm_bytes)


class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2


class ToolResult:
    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        return self.text if isinstance(self.text, str) else json.dumps(self.text)


class Tool:
    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema


class RTToolCall:
    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id


class RTMiddleTier:
    def __init__(self, endpoint: str, deployment: str, credentials: AzureKeyCredential | DefaultAzureCredential, voice_choice: Optional[str] = None):
        self.endpoint = endpoint
        self.deployment = deployment
        self.voice_choice = voice_choice
        self.key = credentials.key if isinstance(credentials, AzureKeyCredential) else None
        self._token_provider = None if self.key else get_bearer_token_provider(
            credentials, "https://cognitiveservices.azure.com/.default")
        self.api_version = "2024-10-01-preview"
        self._tools_pending = {}
        self.tools: dict[str, Tool] = {}
        self.system_message = None
        self.temperature = None
        self.max_tokens = None
        self.disable_audio = None
        self._current_response_transcript = ""
        self._current_response_id = None
        self._last_user_question = None
        self._last_user_timestamp = None
        self._last_user_audio_url = None

        # ✅ Buffer for accumulating raw containerized audio chunks
        self._user_audio_buffer = bytearray()
        # save first bytes for container detection
        self._user_audio_sample = bytearray()
        # partial file path for immediate appended chunks
        self._user_part_path: Optional[str] = None
        # unique id for current utterance (used in part filename)
        self._current_utterance_id: Optional[str] = None

    def _detect_extension_from_bytes(self, sig: bytes) -> str:
        """
        Try to detect a likely file extension from the start of the buffer.
        Checks a few common signatures for webm/ogg/wav/mp3. Fallback '.bin'.
        """
        if not sig:
            print("🔍 DEBUG: Empty signature, defaulting to .bin")
            return ".bin"
        
        print(f"🔍 DEBUG: Detecting extension from {len(sig)} bytes: {sig[:16].hex()}")
        
        # RIFF WAVE
        if len(sig) >= 12 and sig.startswith(b"RIFF") and b"WAVE" in sig[:12]:
            print("🔍 DEBUG: Detected WAV format")
            return ".wav"
        # OggS
        if sig.startswith(b"OggS"):
            print("🔍 DEBUG: Detected OGG format")
            return ".ogg"
        # EBML header for Matroska/WebM: 0x1A45DFA3
        if len(sig) >= 4 and sig[0:4] == b"\x1A\x45\xDF\xA3":
            print("🔍 DEBUG: Detected WebM format")
            return ".webm"
        # ID3 tag (MP3) or frame header 0xFF
        if sig.startswith(b"ID3") or (len(sig) >= 2 and (sig[0] & 0xFF) == 0xFF):
            print("🔍 DEBUG: Detected MP3 format")
            return ".mp3"
        
        print(f"🔍 DEBUG: Unknown format, defaulting to .bin (first 4 bytes: {sig[:4].hex()})")
        return ".bin"

    async def _process_message_to_client(self, msg, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        # msg is an aiohttp WS message from the realtime server
        try:
            payload_text = msg.data
            message = json.loads(payload_text)
        except Exception:
            try:
                # fallback: if msg.data is already a dict-like string
                message = msg.data if isinstance(msg.data, dict) else {}
            except Exception:
                message = {}

        print("⬅️ FROM SERVER:", getattr(msg, "data", "<no data>"))
        try:
            print("🔍 Full transcription message:", json.dumps(message, indent=2))
        except Exception:
            pass

        # Handle completed audio transcription for the user
        if message.get("type") == "conversation.item.input_audio_transcription.completed":
            transcript = message.get("transcript", "")
            print(f"🔍 DEBUG: Transcription completed. Transcript: '{transcript}'")
            print(f"🔍 DEBUG: Buffer size at transcription: {len(self._user_audio_buffer)} bytes")
            print(f"🔍 DEBUG: Sample size: {len(self._user_audio_sample)} bytes")
            
            if self._user_audio_buffer:
                print(f"🔍 DEBUG: First 16 bytes of buffer: {bytes(self._user_audio_buffer[:16]).hex()}")
            else:
                print("🔍 DEBUG: ⚠️ User audio buffer is EMPTY at transcription completion!")

            if transcript:
                print(f"\n👤 USER: {transcript}")
                self._last_user_question = transcript
                self._last_user_timestamp = datetime.utcnow().isoformat()

                # ✅ Save accumulated audio now that transcription is complete
                if self._user_audio_buffer:
                    try:
                        # detect extension from sample or buffer
                        sample = bytes(self._user_audio_sample[:64]) or bytes(self._user_audio_buffer[:64])
                        ext = self._detect_extension_from_bytes(sample)
                        ts = int(datetime.utcnow().timestamp() * 1000)
                        user_filename = os.path.join(LOCAL_AUDIO_DIR, f"user_{ts}{ext}")

                        print(f"🔍 DEBUG: Saving {len(self._user_audio_buffer)} bytes to {user_filename}")

                        # write the accumulated bytes directly (we assume browser sent a container like webm/ogg/wav/mp3)
                        with open(user_filename, "wb") as f:
                            f.write(self._user_audio_buffer)

                        abs_user_filename = os.path.abspath(user_filename)
                        print(f"✅ Saved complete user audio: {abs_user_filename}")

                        # Verify file was actually written
                        if os.path.exists(user_filename):
                            file_size = os.path.getsize(user_filename)
                            print(f"🔍 DEBUG: Verified file exists with size: {file_size} bytes")
                        else:
                            print("🔍 DEBUG: ❌ File was not created!")

                        # remove partial file if exists (we already have final file)
                        if self._user_part_path and os.path.exists(self._user_part_path):
                            try:
                                os.remove(self._user_part_path)
                                print(f"🧹 Removed partial file: {self._user_part_path}")
                            except Exception as e:
                                print("⚠️ Could not remove partial file:", e)
                            finally:
                                self._user_part_path = None
                                self._current_utterance_id = None

                        # Upload to Azure Blob (optional - can be commented out if you only want local saving)
                        try:
                            with open(user_filename, "rb") as data:
                                blob_client = container_client.get_blob_client(blob=os.path.basename(user_filename))
                                blob_client.upload_blob(data, overwrite=True)
                                self._last_user_audio_url = blob_client.url
                                print(f"☁️ Uploaded user audio to Azure: {self._last_user_audio_url}")
                        except Exception as e:
                            print(f"❌ Failed to upload user audio to Azure (local saved OK): {e}")

                        # list folder contents for debugging
                        try:
                            files = os.listdir(LOCAL_AUDIO_DIR)
                            print(f"📂 recorded_audios contents ({len(files)} files): {files}")
                        except Exception as e:
                            print("⚠️ Could not list recorded_audios:", e)

                    except Exception as e:
                        print(f"❌ Failed to save/upload user audio: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # Reset buffers for next utterance
                        print("🔍 DEBUG: Clearing audio buffers for next utterance")
                        self._user_audio_buffer.clear()
                        self._user_audio_sample.clear()
                        self._user_part_path = None
                        self._current_utterance_id = None

                else:
                    print("🔍 DEBUG: ⚠️ No user audio buffer to save at transcription completion - this indicates audio capture failed!")

                # Log
                try:
                    logger.info(f"[{self._last_user_timestamp}] USER AUDIO: {self._last_user_audio_url or 'N/A'}")
                    logger.info(f"[{self._last_user_timestamp}] USER TEXT: {transcript}")
                except Exception as e:
                    print(f"❌ Logging user audio failed: {e}")

        # Handle text inputs from user forwarded by server
        if message.get("type") == "conversation.input":
            user_input = message.get("input", {}).get("content")
            if user_input:
                print(f"📩 USER (text input): {user_input}")
                self._last_user_question = user_input
                self._last_user_timestamp = datetime.utcnow().isoformat()
                self._last_user_audio_url = None  # No audio in this case

        # Model streaming deltas
        if message.get("type") == "response.audio_transcript.delta":
            delta = message.get("delta", "")
            response_id = message.get("response_id", "")

            if response_id != self._current_response_id:
                self._current_response_transcript = ""
                self._current_response_id = response_id

            self._current_response_transcript += delta
            print(f"📨 MODEL (streaming): {self._current_response_transcript}")

        # Model final transcript complete
        if message.get("type") == "response.audio_transcript.done":
            transcript = message.get("transcript", "")
            response_id = message.get("response_id", "")
            final_transcript = transcript or self._current_response_transcript

            if final_transcript:
                print(f"📨 MODEL (complete): {final_transcript}")
                # save model audio to the same folder
                model_filename = os.path.join(LOCAL_AUDIO_DIR, f"model_{int(datetime.utcnow().timestamp()*1000)}.wav")
                model_audio_url = synthesize_and_upload(final_transcript, model_filename)

                try:
                    log_data = [{
                        "question_s": self._last_user_question,
                        "question_audio_url_s": self._last_user_audio_url or "",
                        "answer_s": final_transcript,
                        "answer_audio_url_s": model_audio_url or "",
                        "timestamp": self._last_user_timestamp or datetime.utcnow().isoformat()
                    }]
                    send_log(workspace_id, shared_key, log_type, log_data)
                except Exception as e:
                    print(f"❌ Failed to send log: {e}")

                logger.info(f"[{self._last_user_timestamp}] USER: {self._last_user_question}")
                logger.info(f"[{datetime.utcnow().isoformat()}] MODEL: {final_transcript}")

            self._current_response_transcript = ""
            self._current_response_id = None

        # Return original payload text (forward to client) by default
        return getattr(msg, "data", None)

    async def _process_message_to_server(self, msg, ws: web.WebSocketResponse) -> Optional[str]:
        # msg is an aiohttp WS message from the client
        try:
            payload_text = msg.data
            session_json = json.loads(payload_text)
        except Exception:
            # Not a JSON session update - preserve original
            return getattr(msg, "data", None)

        if session_json.get("type") == "session.update":
            session = session_json.get("session", {})
            if self.system_message:
                session["instructions"] = self.system_message
            if self.temperature is not None:
                session["temperature"] = self.temperature
            if self.max_tokens is not None:
                session["max_response_output_tokens"] = self.max_tokens
            if self.disable_audio is not None:
                session["disable_audio"] = self.disable_audio
            if self.voice_choice:
                session["voice"] = self.voice_choice
            session["tool_choice"] = "auto" if self.tools else "none"
            session["tools"] = [tool.schema for tool in self.tools.values()]
            session["input_audio_transcription"] = {"model": "whisper-1"}
            session_json["session"] = session
            return json.dumps(session_json)

        return getattr(msg, "data", None)

    async def _forward_messages(self, ws: web.WebSocketResponse):
        async with aiohttp.ClientSession(base_url=self.endpoint) as session:
            params = {"api-version": self.api_version, "deployment": self.deployment}
            headers = {"api-key": self.key} if self.key else {
                "Authorization": f"Bearer {self._token_provider()}"}
            if "x-ms-client-request-id" in ws.headers:
                headers["x-ms-client-request-id"] = ws.headers["x-ms-client-request-id"]

            async with session.ws_connect("/openai/realtime", headers=headers, params=params) as target_ws:

                async def from_client_to_server():
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            # Try parsing text JSON first
                            handled = False
                            try:
                                payload = json.loads(msg.data)
                            except Exception:
                                payload = None

                            # Handle input_audio_buffer.append events (this is how user audio is actually sent!)
                            if payload and payload.get("type") == "input_audio_buffer.append":
                                try:
                                    audio_data = payload.get("audio", "")
                                    if audio_data:
                                        # The audio data is base64 encoded in the "audio" field
                                        chunk = base64.b64decode(audio_data)
                                        print(f"🔍 DEBUG: Received input_audio_buffer.append with {len(chunk)} bytes")
                                        if chunk:
                                            print(f"🔍 DEBUG: First 16 bytes of audio buffer chunk: {chunk[:16].hex()}")
                                        
                                        # initialize part file if needed
                                        if not self._current_utterance_id:
                                            self._current_utterance_id = uuid.uuid4().hex
                                            part_name = f"user_{int(datetime.utcnow().timestamp()*1000)}_{self._current_utterance_id}.part"
                                            self._user_part_path = os.path.join(LOCAL_AUDIO_DIR, part_name)
                                            print(f"🟢 Created new partial file: {self._user_part_path}")

                                        # append chunk to partial file
                                        try:
                                            with open(self._user_part_path, "ab") as pf:
                                                pf.write(chunk)
                                            print(f"📥 Appended {len(chunk)} bytes to partial file ({self._user_part_path}).")
                                        except Exception as e:
                                            print("❌ Failed to append to partial file:", e)

                                        # keep a small sample for extension detection later
                                        if not self._user_audio_sample:
                                            self._user_audio_sample.extend(chunk[:128])
                                            print(f"🔍 DEBUG: Captured audio sample of {len(self._user_audio_sample)} bytes")

                                        # keep full buffer for final write on transcription completion
                                        self._user_audio_buffer.extend(chunk)
                                        print(f"📥 Received input_audio_buffer.append (len={len(chunk)}). Buffer size now {len(self._user_audio_buffer)} bytes.")

                                        # Forward the original JSON message to server (not binary)
                                        try:
                                            await target_ws.send_str(msg.data)
                                            handled = True
                                            print("✅ Forwarded input_audio_buffer.append event to server")
                                        except Exception as e:
                                            print("❌ Failed to forward input_audio_buffer.append to server:", e)
                                except (binascii.Error, ValueError) as e:
                                    print("❌ Failed to decode audio data from input_audio_buffer.append:", e)

                            # Legacy support: If client sends base64 audio inside a custom message format
                            elif payload and payload.get("audio_base64"):
                                try:
                                    chunk = base64.b64decode(payload["audio_base64"])
                                    print(f"🔍 DEBUG: Received legacy base64 audio chunk of {len(chunk)} bytes")
                                    if chunk:
                                        print(f"🔍 DEBUG: First 16 bytes of base64 chunk: {chunk[:16].hex()}")
                                    
                                    # initialize part file if needed
                                    if not self._current_utterance_id:
                                        self._current_utterance_id = uuid.uuid4().hex
                                        part_name = f"user_{int(datetime.utcnow().timestamp()*1000)}_{self._current_utterance_id}.part"
                                        self._user_part_path = os.path.join(LOCAL_AUDIO_DIR, part_name)
                                        print(f"🟢 Created new partial file: {self._user_part_path}")

                                    # append chunk to partial file
                                    try:
                                        with open(self._user_part_path, "ab") as pf:
                                            pf.write(chunk)
                                        print(f"📥 Appended {len(chunk)} bytes to partial file ({self._user_part_path}).")
                                    except Exception as e:
                                        print("❌ Failed to append to partial file:", e)

                                    # keep a small sample for extension detection later
                                    if not self._user_audio_sample:
                                        self._user_audio_sample.extend(chunk[:128])
                                        print(f"🔍 DEBUG: Captured audio sample of {len(self._user_audio_sample)} bytes")

                                    # keep full buffer for final write on transcription completion
                                    self._user_audio_buffer.extend(chunk)
                                    print(f"📥 Received base64 chunk (len={len(chunk)}). Buffer size now {len(self._user_audio_buffer)} bytes.")

                                    # Forward binary chunk to the realtime server so transcription works
                                    try:
                                        await target_ws.send_bytes(chunk)
                                        handled = True
                                        print("✅ Forwarded base64 chunk to server")
                                    except Exception as e:
                                        print("❌ Failed to forward decoded base64 chunk to target_ws:", e)
                                except (binascii.Error, ValueError) as e:
                                    print("❌ Failed to decode audio_base64:", e)

                            if handled:
                                continue

                            # Regular text message path
                            try:
                                new_msg = await self._process_message_to_server(msg, ws)
                                if new_msg is not None:
                                    await target_ws.send_str(new_msg)
                            except Exception as e:
                                print("Failed to process client message:", e)

                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            # Accumulate chunks, append to a .part file for immediate local recording,
                            # and forward them to the realtime server.
                            try:
                                chunk = msg.data
                                print(f"🔍 DEBUG: Received binary audio chunk of {len(chunk)} bytes")
                                if chunk:
                                    print(f"🔍 DEBUG: First 16 bytes of binary chunk: {chunk[:16].hex()}")
                                    
                                    # initialize an utterance id / part file if needed
                                    if not self._current_utterance_id:
                                        # unique id so simultaneous connections don't clobber each other
                                        self._current_utterance_id = uuid.uuid4().hex
                                        part_name = f"user_{int(datetime.utcnow().timestamp()*1000)}_{self._current_utterance_id}.part"
                                        self._user_part_path = os.path.join(LOCAL_AUDIO_DIR, part_name)
                                        print(f"🟢 Created new partial file: {self._user_part_path}")

                                    # append chunk to partial file immediately (so it's available on disk)
                                    try:
                                        with open(self._user_part_path, "ab") as pf:
                                            pf.write(chunk)
                                        print(f"📥 Appended {len(chunk)} bytes to partial file ({self._user_part_path}).")
                                    except Exception as e:
                                        print("❌ Failed to append to partial file:", e)

                                    # keep a small sample for extension detection later
                                    if not self._user_audio_sample:
                                        self._user_audio_sample.extend(chunk[:128])
                                        print(f"🔍 DEBUG: Captured audio sample of {len(self._user_audio_sample)} bytes")

                                    # keep full buffer for final write on transcription completion
                                    self._user_audio_buffer.extend(chunk)
                                    print(f"📥 Received binary audio chunk (len={len(chunk)}). Buffer size now {len(self._user_audio_buffer)} bytes.")

                                # Forward binary chunk so transcription still works
                                try:
                                    await target_ws.send_bytes(chunk)
                                    print("✅ Forwarded binary chunk to server")
                                except Exception as e:
                                    print("❌ Failed to forward binary chunk to target_ws:", e)
                            except Exception as e:
                                print("❌ Failed to handle/forward binary audio chunk:", e)
                                # fallback: try forwarding raw data
                                try:
                                    await target_ws.send_bytes(msg.data)
                                    print("⚠️ Used fallback forwarding for binary data")
                                except Exception as e2:
                                    print("❌ Failed to forward binary to server as fallback:", e2)

                        else:
                            print("⚠️ Unexpected message type from client:", msg.type)

                    if target_ws:
                        print("🔌 Closing OpenAI's realtime socket connection.")
                        await target_ws.close()

                async def from_server_to_client():
                    try:
                        async for msg in target_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                new_msg = await self._process_message_to_client(msg, ws, target_ws)
                                if new_msg and not ws.closed:
                                    await ws.send_str(new_msg)
                            else:
                                print("Unexpected message type from server:", msg.type)
                    except Exception as e:
                        print("Server WebSocket closed with exception:", e)

                try:
                    await asyncio.gather(from_client_to_server(), from_server_to_client())
                except ConnectionResetError:
                    print("🔌 Connection reset")

    async def _websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self._forward_messages(ws)
        return ws

    def attach_to_app(self, app: web.Application, path: str):
        """
        Attach the realtime ws endpoint and an /files endpoint to inspect saved audio files.
        """
        app.router.add_get(path, self._websocket_handler)
        app.router.add_get("/files", list_audio_files)
        print(f"✅ Attached realtime websocket at {path} and file listing at /files")


# Helper endpoint to list saved audio files
async def list_audio_files(request):
    files = []
    try:
        for fn in os.listdir(LOCAL_AUDIO_DIR):
            path = os.path.join(LOCAL_AUDIO_DIR, fn)
            try:
                st = os.stat(path)
                files.append({
                    "name": fn,
                    "size": st.st_size,
                    "mtime": datetime.utcfromtimestamp(st.st_mtime).isoformat()
                })
            except Exception:
                files.append({"name": fn, "size": None, "mtime": None})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(files)