import edge_tts
import tempfile
import os

VOICE = "en-US-AriaNeural"  # High quality, free Microsoft voice

async def generate_audio(text: str) -> str:
    """Generates MP3 audio and returns the file path."""
    communicate = edge_tts.Communicate(text, VOICE)
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    await communicate.save(temp_file.name)
    return temp_file.name