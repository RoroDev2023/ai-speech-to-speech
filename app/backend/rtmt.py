import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Optional
from datetime import datetime, timezone
from log import send_log, workspace_id, shared_key, log_type


import aiohttp
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

print("✅ RTMT MODULE LOADED")

# Set up logging
logger = logging.getLogger("voicerag")
logging.basicConfig(
    filename="chat_history.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log_interaction(role: str, content: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    # Log locally
    logger.info(f"[{timestamp}] {role.upper()}: {content}")

    # Log to Azure
    try:
        log_data = [{
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }]
        send_log(workspace_id, shared_key, log_type, log_data)
    except Exception as e:
        print(f"❌ Failed to send Azure log: {e}")


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
        self._token_provider = None if self.key else get_bearer_token_provider(credentials, "https://cognitiveservices.azure.com/.default")
        self.api_version = "2024-10-01-preview"
        self._tools_pending = {}
        self.tools: dict[str, Tool] = {}
        self.system_message = None
        self.temperature = None
        self.max_tokens = None
        self.disable_audio = None
        self._current_response_transcript = ""
        self._current_response_id = None

    async def _process_message_to_client(self, msg: str, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        message = json.loads(msg.data)
        print("⬅️ FROM SERVER:", msg.data)

        if message.get("type") == "conversation.item.created":
            item = message.get("item", {})
            if item.get("type") == "message" and item.get("role") == "user":
                # Check for audio transcription in user message
                content = item.get("content", [])
                for content_part in content:
                    if content_part.get("type") == "input_text":
                        transcript = content_part.get("text")
                        if transcript:
                            print(f"📩 USER (speech transcribed): {transcript}")
                            log_interaction("user", transcript)
                    elif content_part.get("type") == "input_audio":
                        transcript = content_part.get("transcript")
                        if transcript and transcript != "null":
                            print(f"📩 USER (audio transcript): {transcript}")
                            log_interaction("user", transcript)
                        else:
                            print("🎤 USER: [Audio received, waiting for transcription...]")

        # MAIN USER SPEECH TRANSCRIPTION - This is the correct event type for Azure OpenAI
        if message.get("type") == "conversation.item.input_audio_transcription.completed":
            transcript = message.get("transcript", "")
            item_id = message.get("item_id", "")
            if transcript:
                print(f"\n👤 USER: {transcript}")
                log_interaction("user", transcript)

        if message.get("type") == "conversation.input":
            user_input = message.get("input", {}).get("content")
            if user_input:
                print(f"📩 USER (conversation input): {user_input}")
                log_interaction("user", user_input)

        if message.get("type") == "response.audio_transcript.delta":
            delta = message.get("delta", "")
            response_id = message.get("response_id", "")
            
            if response_id != self._current_response_id:
                self._current_response_transcript = ""
                self._current_response_id = response_id
            
            self._current_response_transcript += delta
            print(f"📨 MODEL (streaming): {self._current_response_transcript}")

        if message.get("type") == "response.audio_transcript.done":
            transcript = message.get("transcript", "")
            response_id = message.get("response_id", "")
            
            final_transcript = transcript or self._current_response_transcript
            if final_transcript:
                print(f"📨 MODEL (complete): {final_transcript}")
                log_interaction("model", final_transcript)
            
            self._current_response_transcript = ""
            self._current_response_id = None

        known_types = {
            "response.audio.done", "response.audio_transcript.delta", "response.content_part.done",
            "response.audio.delta", "session.created", "session.updated", "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "response.created",
            "rate_limits.updated", "conversation.item.created", "response.output_item.added", "response.content_part.added",
            "conversation.item.input_audio_transcription.completed", "response.audio_transcript.done"
        }
        
        if message.get("type") not in known_types:
            print(f"🔍 UNKNOWN SERVER MESSAGE TYPE: {message.get('type')}")

        return msg.data

    async def _process_message_to_server(self, msg: str, ws: web.WebSocketResponse) -> Optional[str]:
        message = json.loads(msg.data)

        # Apply session configuration if needed
        if message.get("type") == "session.update":
            session = message["session"]
            if self.system_message: 
                session["instructions"] = self.system_message
            if self.temperature: 
                session["temperature"] = self.temperature
            if self.max_tokens: 
                session["max_response_output_tokens"] = self.max_tokens
            if self.disable_audio is not None: 
                session["disable_audio"] = self.disable_audio
            if self.voice_choice: 
                session["voice"] = self.voice_choice
            session["tool_choice"] = "auto" if self.tools else "none"
            session["tools"] = [tool.schema for tool in self.tools.values()]
            
            # Enable input audio transcription to get user speech transcripts
            session["input_audio_transcription"] = {
                "model": "whisper-1"
            }
            
            return json.dumps(message)

        return msg.data

    async def _forward_messages(self, ws: web.WebSocketResponse):
        async with aiohttp.ClientSession(base_url=self.endpoint) as session:
            params = {"api-version": self.api_version, "deployment": self.deployment}
            headers = {"api-key": self.key} if self.key else {"Authorization": f"Bearer {self._token_provider()}"}
            if "x-ms-client-request-id" in ws.headers:
                headers["x-ms-client-request-id"] = ws.headers["x-ms-client-request-id"]

            async with session.ws_connect("/openai/realtime", headers=headers, params=params) as target_ws:

                async def from_client_to_server():
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                new_msg = await self._process_message_to_server(msg, ws)
                                if new_msg is not None:
                                    await target_ws.send_str(new_msg)
                            except Exception as e:
                                print("Failed to process client message:", e)
                        else:
                            print("Error: unexpected message type from client:", msg.type)

                    if target_ws:
                        print("Closing OpenAI's realtime socket connection.")
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

    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)