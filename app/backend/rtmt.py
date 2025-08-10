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
# from log import send_log, workspace_id, shared_key, log_type # You'll need to uncomment this if you use the logging function
import wave
import os
import uuid
import binascii

import aiohttp
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# You must have your logger file (`log.py`) in the same directory as this script.
from log import send_log, workspace_id, shared_key, log_type

print("✅ RTMT MODULE LOADED")

# Setup local log file
logger = logging.getLogger("voicerag")
logging.basicConfig(
    filename="chat_history.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Azure credentials (as provided in your snippet)
SPEECH_KEY=your_speech_key_here
SPEECH_REGION=your_region_here
BLOB_CONNECTION_STRING=your_blob_connection_string_here
BLOB_CONTAINER=your_container_name_here


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
    """
    Saves a byte array of raw PCM audio data into a valid WAV file.
    A small buffer of silence is added at the beginning to prevent truncation.
    """
    # Define audio parameters
    sample_rate = 24000
    sample_width = 2
    num_channels = 1
    
    # Create 500ms of silence
    silence_duration_ms = 500
    silence_frames = int(sample_rate * silence_duration_ms / 1000)
    silence_bytes = b'\x00' * silence_frames * sample_width * num_channels
    
    # Prepend silence to the actual audio data
    buffered_pcm_bytes = silence_bytes + pcm_bytes

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(buffered_pcm_bytes)


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

        # Buffer for accumulating user audio chunks
        self._user_audio_buffer = bytearray()
        # flag to know when an utterance has started
        self._is_recording_user_audio = False

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

        # Handle user audio transcription events
        if message.get("type") == "conversation.item.input_audio_transcription.completed":
            transcript = message.get("transcript", "")
            print(f"🔍 DEBUG: Transcription completed. Transcript: '{transcript}'")

            if transcript and self._is_recording_user_audio:
                print(f"\n👤 USER: {transcript}")
                self._last_user_question = transcript
                self._last_user_timestamp = datetime.utcnow().isoformat()

                # Always save the buffered audio as a WAV file
                if self._user_audio_buffer:
                    try:
                        ts = int(datetime.utcnow().timestamp() * 1000)
                        user_filename = os.path.join(LOCAL_AUDIO_DIR, f"user_{ts}.wav")
                        
                        # Use the save_pcm_to_wav function to correctly format the buffered data
                        save_pcm_to_wav(self._user_audio_buffer, user_filename)
                        
                        abs_user_filename = os.path.abspath(user_filename)
                        print(f"✅ Saved complete user audio: {abs_user_filename}")

                        # Upload to Azure Blob
                        try:
                            with open(user_filename, "rb") as data:
                                blob_client = container_client.get_blob_client(blob=os.path.basename(user_filename))
                                blob_client.upload_blob(data, overwrite=True)
                                self._last_user_audio_url = blob_client.url
                                print(f"☁️ Uploaded user audio to Azure: {self._last_user_audio_url}")
                        except Exception as e:
                            print(f"❌ Failed to upload user audio to Azure (local saved OK): {e}")

                    except Exception as e:
                        print(f"❌ Failed to save/upload user audio: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # Reset buffers and flag for the next utterance
                        print("🔍 DEBUG: Clearing audio buffers for next utterance")
                        self._user_audio_buffer.clear()
                        self._is_recording_user_audio = False

                else:
                    print("🔍 DEBUG: ⚠️ No user audio buffer to save at transcription completion.")

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
                    print("✅ Log sent successfully to Azure workspace.")
                except Exception as e:
                    print(f"❌ Failed to send log to Azure: {e}")

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

                            # Handle input_audio_buffer.append events
                            if payload and payload.get("type") == "input_audio_buffer.append":
                                try:
                                    audio_data = payload.get("audio", "")
                                    if audio_data:
                                        chunk = base64.b64decode(audio_data)
                                        self._user_audio_buffer.extend(chunk)
                                        self._is_recording_user_audio = True
                                        print(f"📥 Received input_audio_buffer.append (len={len(chunk)}). Buffer size now {len(self._user_audio_buffer)} bytes.")

                                        # Forward the original JSON message to server
                                        await target_ws.send_str(msg.data)
                                        handled = True
                                except (binascii.Error, ValueError) as e:
                                    print("❌ Failed to decode audio data from input_audio_buffer.append:", e)
                                
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
                            # This path handles clients sending raw binary audio chunks
                            try:
                                chunk = msg.data
                                print(f"🔍 DEBUG: Received binary audio chunk of {len(chunk)} bytes")
                                
                                # Append the chunk to the buffer
                                self._user_audio_buffer.extend(chunk)
                                self._is_recording_user_audio = True
                                print(f"📥 Received binary audio chunk (len={len(chunk)}). Buffer size now {len(self._user_audio_buffer)} bytes.")

                                # Forward the binary chunk to the server
                                await target_ws.send_bytes(chunk)
                                print("✅ Forwarded binary chunk to server")
                            except Exception as e:
                                print("❌ Failed to handle/forward binary audio chunk:", e)
                        
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