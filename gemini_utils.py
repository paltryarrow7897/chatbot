from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from gradio import ChatMessage, Error, MultimodalTextbox
import hashlib
from pathlib import Path

load_dotenv(override=True)

GEMINI_FILE_TYPES = [".pdf", ".js", ".py", ".txt", ".html", ".css", ".md", ".csv", ".xml", ".rtf"]

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def generate_file_name(file_path: str) -> str:
    hash_bytes = hashlib.sha256(file_path.encode()).digest()
    hash_int = int.from_bytes(hash_bytes, "big")
    base36_chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    base36_id = ""
    num = hash_int
    while num:
        base36_id = base36_chars[num % 36] + base36_id
        num //= 36
    base36_id = base36_id[:8]
    formatted_id = f"{base36_id[:4]}-{base36_id[4:8]}"
    return formatted_id


def file_exists(name: str) -> bool:
    existing_files = []
    for f in client.files.list():
        existing_files.append(f.name.split("/")[-1])
    if name in existing_files:
        return True
    return False


def upload_files(file_paths: List[Path]) -> List[str]:
    file_names = []
    for path in file_paths:
        # Only lowercase alphanumeric characters and dashes allowed, generating a unique 8-character file name
        file_name = generate_file_name(file_path=str(path))
        if not file_exists(name=file_name):
            file = client.files.upload(file=path, config={"name": file_name, "display_name": path.name})
            while file.state.name == "PROCESSING":
                time.sleep(2)
                file = client.files.get(name=file_name)
        file_names.append(file_name)
    return file_names


def _get_content(_message: Dict[str, Any]) -> types.Content|types.File|None:
    message_role = "model" if _message["role"] == "assistant" else "user"

    if isinstance(_message["content"], tuple):
        path = Path(_message["content"][0])
        files = upload_files([path])
        content = client.files.get(name=files[0])

    elif isinstance(_message["content"], str):
        content = types.Content(role=message_role, parts=[types.Part.from_text(text=_message["content"])])

    else:
        content = None

    return content


def chat(model_name: str, _history: List[ChatMessage], query: MultimodalTextbox) -> List[ChatMessage]|None:
    contents = []

    for message in _history:
        content = _get_content(message)
        if content is not None:
            contents.append(content)
        
    _history.extend([ChatMessage(role="user", content={"path": file}) for file in query["files"]])
    
    files = upload_files([Path(i) for i in query["files"]])
    if files:
        contents.extend([client.files.get(name=file) for file in files])

    contents.append(types.UserContent(parts=types.Part.from_text(text=query["text"])))
    _history.append(ChatMessage(role="user", content=query["text"]))

    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=contents
        )
        _history.append(ChatMessage(role="assistant", content=response.text))
        return _history
    
    except Exception as e:
        raise Error(message=str(e))
