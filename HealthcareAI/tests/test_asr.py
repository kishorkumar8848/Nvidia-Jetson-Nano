import os
import pytest
import numpy as np
import scipy.io.wavfile as wav
from fastapi.testclient import TestClient
from backend.app import app
from config.config import settings
from asr.audio_recorder import AudioRecorder
from asr.asr_service import ASRService
from rag.rag_service import RAGService

client = TestClient(app)

@pytest.fixture
def temp_silent_wav():
    """Generates a temporary silent WAV file for transcribing and cleans it up."""
    logs_dir = settings.get_absolute_path("logs")
    os.makedirs(logs_dir, exist_ok=True)
    file_path = os.path.join(logs_dir, "temp_test_silent.wav")
    
    # Write a silent 16kHz WAV file of 1 second duration
    sample_rate = 16000
    silent_data = np.zeros(sample_rate, dtype=np.int16)
    wav.write(file_path, sample_rate, silent_data)
    
    yield file_path
    
    if os.path.exists(file_path):
        os.remove(file_path)

def test_audio_recorder_write():
    """Verify that AudioRecorder saves WAV files to disk even if soundcard is missing (via silent fallback)."""
    logs_dir = settings.get_absolute_path("logs")
    output_path = os.path.join(logs_dir, "test_record_class.wav")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Record a very short 0.1s block
    success = AudioRecorder.record_audio(output_path, duration=0.1)
    assert success is True
    assert os.path.exists(output_path)
    
    # Check WAV is readable
    sr, data = wav.read(output_path)
    assert sr == 16000
    assert len(data) > 0
    
    os.remove(output_path)

def test_asr_service_mock_languages(temp_silent_wav):
    """Verify that ASRService correctly returns the target local languages mock strings in fallback mode."""
    asr_service = ASRService()
    
    # 1. Tamil transcribing
    tamil_txt = asr_service.transcribe(temp_silent_wav, language="ta", force_mock=True)
    assert tamil_txt == "எனக்கு காய்ச்சல் மற்றும் தலைவலி உள்ளது"
    
    # 2. Hindi transcribing
    hindi_txt = asr_service.transcribe(temp_silent_wav, language="hi", force_mock=True)
    assert hindi_txt == "मुझे बुखार और सिरदर्द है"
    
    # 3. English transcribing
    english_txt = asr_service.transcribe(temp_silent_wav, language="en", force_mock=True)
    assert english_txt == "Patient has fever and headache."

def test_backend_audio_upload_pipeline(temp_silent_wav):
    """Test full integration: uploading a WAV audio file to the API endpoint performs ASR and queries LLM."""
    import backend.routes.assistant
    
    # Save original engines and force mock fallbacks
    orig_asr_engine = backend.routes.assistant.asr_service.asr_engine
    orig_nmt_engine = backend.routes.assistant.translation_service.nmt_engine
    orig_llm_running = backend.routes.assistant.llm_service.is_ollama_running
    
    backend.routes.assistant.asr_service.asr_engine = None
    backend.routes.assistant.translation_service.nmt_engine = None
    backend.routes.assistant.llm_service.is_ollama_running = lambda: False

    # Write a protocol for fever treatment so RAG doesn't return empty warning
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    os.makedirs(protocols_dir, exist_ok=True)
    test_protocol_path = os.path.join(protocols_dir, "test_temp_fever.txt")
    with open(test_protocol_path, "w", encoding="utf-8") as f:
        f.write("Clinical guidelines: For high fever and headache, administer Paracetamol 500mg.")

    try:
        # Upload the silent WAV as audio_file
        with open(temp_silent_wav, "rb") as f:
            files = {"audio_file": ("audio.wav", f, "audio/wav")}
            data = {"language": "ta"}  # Tamil lang
            
            response = client.post("/api/assistant/interact", files=files, data=data)
            
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["success"] is True
        assert json_data["input_type"] == "audio"
        assert json_data["detected_text"] == "எனக்கு காய்ச்சல் மற்றும் தலைவலி உள்ளது"
        assert json_data["translated_query"] == "fever and headache"
        assert "How many days have you had the fever" in json_data["response_en"]
        
    finally:
        # Restore original engines
        backend.routes.assistant.asr_service.asr_engine = orig_asr_engine
        backend.routes.assistant.translation_service.nmt_engine = orig_nmt_engine
        backend.routes.assistant.llm_service.is_ollama_running = orig_llm_running
        
        # Cleanup test protocol
        if os.path.exists(test_protocol_path):
            os.remove(test_protocol_path)
        # Clear vector store cache
        rag_service = RAGService()
        rag_service.check_and_rebuild_index()
