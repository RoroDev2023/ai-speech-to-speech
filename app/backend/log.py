import json
import requests
import hashlib
import hmac
import base64
from datetime import datetime

# --- Azure Log Config ---
WORKSPACE_ID=your_workspace_id_here
SHARED_KEY=your_shared_key_here
LOG_TYPE=VoiceRAGLogs_Custom


def build_signature(customer_id, shared_key, date, content_length, method, content_type, resource):
    x_headers = f'x-ms-date:{date}'
    string_to_hash = f'{method}\n{content_length}\n{content_type}\n{x_headers}\n{resource}'
    bytes_to_hash = bytes(string_to_hash, encoding='utf-8')
    decoded_key = base64.b64decode(shared_key)
    encoded_hash = hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
    encoded_hash = base64.b64encode(encoded_hash).decode()
    authorization = f'SharedKey {customer_id}:{encoded_hash}'
    return authorization

def send_log(workspace_id, shared_key, log_type, log_data):
    body = json.dumps(log_data)
    method = 'POST'
    content_type = 'application/json'
    resource = '/api/logs'
    rfc1123date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    content_length = len(body)

    signature = build_signature(workspace_id, shared_key, rfc1123date, content_length, method, content_type, resource)

    uri = f'https://{workspace_id}.ods.opinsights.azure.com{resource}?api-version=2016-04-01'

    headers = {
        'Content-Type': content_type,
        'Authorization': signature,
        'Log-Type': log_type,
        'x-ms-date': rfc1123date,
        'time-generated-field': 'timestamp'
    }

    response = requests.post(uri, data=body, headers=headers)
    if 200 <= response.status_code < 300:
        print('✅ Log uğurla göndərildi!')
    else:
        print(f'❌ Log göndərilmədi: {response.status_code} - {response.text}')
