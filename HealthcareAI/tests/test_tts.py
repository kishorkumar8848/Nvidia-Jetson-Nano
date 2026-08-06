import os
import pytest
import scipy.io.wavfile as wav
from fastapi.testclient import TestClient
from backend.app import app
from config.config import settings
from tts.tts_service import TTSService

client = TestClient(app)

def test_tts_service_synthesis_en():
    """Verify that TTSService compiles English text into speech WAV files."""
    logs_dir = settings.get_absolute_path("logs")
    output_path = os.path.join(logs_dir, "test_tts_output_en.wav")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    service = TTSService()
    success = service.synthesize("Administer 500 milligrams paracetamol.", "en", output_path)
    assert success is True
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Read WAV metadata to verify format
    sr, data = wav.read(output_path)
    assert sr > 0
    assert len(data) > 0
    
    os.remove(output_path)

def test_tts_service_synthesis_tamil():
    """Verify Tamil text compiles into speech audio file (via local engine or mock beep fallback)."""
    logs_dir = settings.get_absolute_path("logs")
    output_path = os.path.join(logs_dir, "test_tts_output_ta.wav")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    service = TTSService()
    success = service.synthesize("பாராசிட்டமால் மாத்திரை வழங்கவும்.", "ta", output_path)
    assert success is True
    assert os.path.exists(output_path)
    
    os.remove(output_path)

def test_backend_endpoint_tts_generation():
    """Test full integration: querying assistant generates TTS WAV and hosts it statically for download."""
    data = {
        "text_input": "My baby has a high fever.",
        "language": "ta"
    }
    
    response = client.post("/api/assistant/interact", data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    
    audio_url = json_data.get("audio_response_url")
    audio_path = json_data.get("audio_response_path")
    assert audio_url is not None
    assert audio_path is not None
    assert "/audio/" in audio_url
    
    # Verify static file download
    down_response = client.get(audio_url)
    assert down_response.status_code == 200
    assert len(down_response.content) > 0
    
    # Cleanup generated file
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
