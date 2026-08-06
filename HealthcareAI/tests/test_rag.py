import os
import shutil
import pytest
from config.config import settings
from rag.document_loader import DocumentLoader
from rag.text_splitter import ProtocolTextSplitter
from rag.rag_service import RAGService

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_protocols():
    """Sets up temporary test files in data/protocols and cleans them up after tests run."""
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    os.makedirs(protocols_dir, exist_ok=True)

    # File paths
    fever_path = os.path.join(protocols_dir, "test_fever.txt")
    cough_path = os.path.join(protocols_dir, "test_cough.txt")

    # Write test content
    with open(fever_path, "w", encoding="utf-8") as f:
        f.write("Fever Treatment Protocol: For high fever, apply cold compress and administer Paracetamol 500mg every 6 hours.")
    
    with open(cough_path, "w", encoding="utf-8") as f:
        f.write("Cough and Cold Guidelines: Suggest warm fluids, avoid cold water, and prescribe cough syrup for adults if cough is dry.")

    yield  # Runs the tests

    # Teardown: Remove test files
    for filename in ["test_fever.txt", "test_cough.txt", "test_malaria.txt"]:
        path = os.path.join(protocols_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    # Force a final rebuild to clear vector store of test documents
    rag_service = RAGService()
    rag_service.check_and_rebuild_index()

def test_document_loader():
    """Verify that DocumentLoader correctly reads files from protocols directory."""
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    docs = DocumentLoader.load_directory(protocols_dir)
    assert len(docs) >= 2
    
    # Assert contents are present in loaded docs
    contents = [doc.page_content for doc in docs]
    assert any("Fever Treatment Protocol" in content for content in contents)
    assert any("Cough and Cold Guidelines" in content for content in contents)

def test_text_splitter():
    """Verify that ProtocolTextSplitter chunks long documents."""
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    docs = DocumentLoader.load_directory(protocols_dir)
    
    splitter = ProtocolTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    assert len(chunks) > len(docs)  # Should split into more chunks due to small chunk size

def test_rag_retrieval_and_auto_rebuild():
    """Verify RAGService indexes, queries, and automatically detects file changes."""
    rag_service = RAGService()
    
    # Force rebuild with test files
    rebuilt = rag_service.check_and_rebuild_index()
    assert rebuilt is True or os.path.exists(os.path.join(rag_service.vector_store_dir, "index.faiss"))
    
    # Query for fever
    fever_results = rag_service.retrieve_context("paracetamol fever", top_k=1)
    assert len(fever_results) == 1
    assert "Paracetamol" in fever_results[0]["content"]
    assert "test_fever.txt" in fever_results[0]["metadata"]["source"]

    # Query for cough
    cough_results = rag_service.retrieve_context("dry cough syrup", top_k=1)
    assert len(cough_results) == 1
    assert "cough syrup" in cough_results[0]["content"]
    
    # --- Test Auto-rebuild mechanism ---
    # Add a new file programmatically
    protocols_dir = settings.get_absolute_path(settings.PROTOCOLS_DIR)
    malaria_path = os.path.join(protocols_dir, "test_malaria.txt")
    with open(malaria_path, "w", encoding="utf-8") as f:
        f.write("Malaria Diagnostic Protocol: check blood smear for Plasmodium falciparum malaria parasite.")

    # Execute search query; should automatically trigger rebuild and retrieve new document
    malaria_results = rag_service.retrieve_context("plasmodium falciparum malaria", top_k=1)
    assert len(malaria_results) == 1
    assert "Plasmodium falciparum" in malaria_results[0]["content"]
