import os
import io
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from config.config import settings

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns instructions to docs."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "docs_url" in json_data
    assert json_data["docs_url"] == "/docs"

def test_status_endpoint():
    """Test standard health status check and configured models representation."""
    response = client.get("/api/status")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "configurations" in json_data
    assert json_data["configurations"]["llm_model"] == settings.LLM_MODEL

def test_assistant_interact_success():
    """Test assistant interaction endpoint with valid Form data."""
    data = {
        "text_input": "Hello patient feels feverish",
        "language": "ta"
    }
    response = client.post("/api/assistant/interact", data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["input_type"] == "text"
    assert "response_local" in json_data
    assert "audio_response_path" in json_data

def test_assistant_interact_missing_fields():
    """Test assistant endpoint validation when both audio and text are missing."""
    data = {
        "language": "en"
    }
    response = client.post("/api/assistant/interact", data=data)
    assert response.status_code == 400
    json_data = response.json()
    assert json_data["success"] is False
    assert "detail" in json_data

def test_protocols_upload_invalid_type():
    """Test protocol upload endpoint blocks non-allowed extensions (e.g. png)."""
    file_content = b"fake image bytes"
    file = {"file": ("test_image.png", io.BytesIO(file_content), "image/png")}
    response = client.post("/api/protocols/upload", files=file)
    assert response.status_code == 400
    assert "Only PDF and TXT files are allowed" in response.json()["detail"]

def test_protocols_upload_success():
    """Test protocol upload successfully accepts PDF or TXT file."""
    file_content = b"Standard Clinical Guideline Content"
    file = {"file": ("test_guideline.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/protocols/upload", files=file)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["filename"] == "test_guideline.txt"
    
    # Cleanup uploaded test file if it was saved
    saved_path = json_data.get("saved_path")
    if saved_path and os.path.exists(saved_path):
        os.remove(saved_path)

def test_protocols_rebuild():
    """Test the vector database rebuild endpoint trigger."""
    response = client.post("/api/protocols/rebuild")
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_history_endpoint():
    """Test retrieving conversation logs."""
    response = client.get("/api/history")
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert len(json_data) > 0
    assert "user_query_local" in json_data[0]
