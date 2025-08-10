from azure.storage.blob import BlobServiceClient
import os

container_name = os.getenv("BLOB_CONTAINER_NAME")
connect_str = os.getenv("BLOB_CONNECTION_STRING")

def upload_audio_to_blob(filepath: str) -> str:
    if not container_name or not connect_str:
        raise RuntimeError("Missing BLOB_CONTAINER_NAME or BLOB_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service.get_blob_client(container=container_name, blob=os.path.basename(filepath))
    with open(filepath, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    return blob_client.url
