from openai import OpenAI
from io import BytesIO
from dotenv import load_dotenv
import os
from typing import Literal


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

FILE_TYPES = Literal["TEXT", "AUDIO", "IMAGE", "PDF", "EXCEL"]


def translate_openai(client: OpenAI, audio: BytesIO, output_language: str|None = "EN"):
    translation = client.audio.translations.create(
        file=audio,
        model="whisper-1",
        response_format="text"
    )
    if output_language == "EN":
        return translation
    else:
        translation = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "user", "content": f"Translate this text into {output_language}."},
                {"role": "user", "content": translation}
            ]
        )
        return translation.choices[0].message.content


def request_openai(
        client: OpenAI, 
        file_type: FILE_TYPES, 
        file: BytesIO|None,
        text: str|None
    ):

    if file is None and text is None:
        raise ValueError("Please write a message or upload a file first.")

    if file_type == "TEXT":
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            temperature=0, 
            messages=[
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    
    # if file_type == "IMAGE":
    #     response = 
