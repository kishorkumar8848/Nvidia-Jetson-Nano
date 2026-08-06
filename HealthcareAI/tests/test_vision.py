import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app
from config.config import settings
from vision.camera_capture import CameraCapture
from vision.vision_service import VisionService
from rag.rag_service import RAGService

client = TestClient(app)

@pytest.fixture
def temp_rash_image():
    """Generates a pink/red simulated rash JPEG placeholder and deletes it after test."""
    logs_dir = settings.get_absolute_path("logs")
    os.makedirs(logs_dir, exist_ok=True)
    file_path = os.path.join(logs_dir, "temp_test_rash.jpg")
    
    # Generate placeholder
    success = CameraCapture._generate_placeholder_image(file_path)
    assert success is True
    assert os.path.exists(file_path)
    
    yield file_path
    
    if os.path.exists(file_path):
        os.remove(file_path)

def test_camera_capture_saved():
    """Verify that CameraCapture successfully writes frame output (via hardware capture or PIL fallback)."""
    logs_dir = settings.get_absolute_path("logs")
    output_path = os.path.join(logs_dir, "test_cam_capture.jpg")
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    success = CameraCapture.capture_frame(output_path, camera_index=0)
    assert success is True
    assert os.path.exists(output_path)
    
    # Confirm it's a valid image
    with Image.open(output_path) as img:
        assert img.format == "JPEG"
        
    os.remove(output_path)

def test_vision_service_pixel_reasoning(temp_rash_image):
    """Verify that VisionService correctly analyzes image red channels to simulate VLM rash diagnosis."""
    vision_service = VisionService()
    
    # Run analysis on simulated red rash image
    rash_desc = vision_service.extract_symptoms(temp_rash_image, force_mock=True)
    assert "rash" in rash_desc
    assert "inflammation" in rash_desc or "red" in rash_desc

def test_assistant_endpoint_with_image_upload(temp_rash_image):
    """Test full pipeline integration: uploading a symptom image extracts visual data and shapes LLM answers."""
    # Write a dummy protocol addressing skin rashes
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    os.makedirs(protocols_dir, exist_ok=True)
    test_protocol_path = os.path.join(protocols_dir, "test_rash_protocol.txt")
    with open(test_protocol_path, "w", encoding="utf-8") as f:
        f.write("Baby Fever and Skin Rash Protocol: For any baby, child, or patient presenting with high fever and skin rash, refer immediately if rash or stiff neck is present. Otherwise, administer Paracetamol.")

    try:
        # Rebuild RAG Vector Store to read this protocol
        rag_service = RAGService()
        rag_service.check_and_rebuild_index()

        with open(temp_rash_image, "rb") as img_file:
            files = {"image_file": ("rash.jpg", img_file, "image/jpeg")}
            data = {
                "text_input": "My baby has a high fever.",
                "language": "en"
            }
            response = client.post("/api/assistant/interact", files=files, data=data)

        assert response.status_code == 200
        json_data = response.json()
        assert json_data["success"] is True
        assert "rash" in json_data["extracted_symptoms"]
        assert "refer immediately" in json_data["response_en"] or "Paracetamol" in json_data["response_en"]

    finally:
        # Cleanup
        if os.path.exists(test_protocol_path):
            os.remove(test_protocol_path)
        rag_service = RAGService()
        rag_service.check_and_rebuild_index()
