import pytest
from llm.prompt_templates import format_prompt, CLINICAL_SYSTEM_PROMPT
from llm.llm_service import LLMService

def test_prompt_formatting():
    """Verify that formatting prompts maps context and query strings correctly."""
    context = ["Protocol A: Give paracetamol for fever.", "Protocol B: If rash, refer to hospital."]
    question = "How to treat fever?"
    
    prompt = format_prompt(question, context)
    assert CLINICAL_SYSTEM_PROMPT in prompt
    assert "Protocol A: Give paracetamol for fever." in prompt
    assert "Protocol B: If rash, refer to hospital." in prompt
    assert "Clinical Question: How to treat fever?" in prompt

def test_llm_service_offline_mock_fallback():
    """Verify that LLMService falls back to structured mock outputs when force_mock=True."""
    llm_service = LLMService()
    
    # 1. Test query for fever with paracetamol context
    context = ["For fever, administer Paracetamol 500mg."]
    ans = llm_service.generate_answer("What should we do for high fever?", context, force_mock=True)
    assert "Paracetamol" in ans
    assert "[MOCK LLM RESPONSE]" in ans

    # 2. Test query for cough with cough syrup context
    context = ["Suggest warm fluids, and prescribe cough syrup for dry cough."]
    ans = llm_service.generate_answer("How to treat a dry cough?", context, force_mock=True)
    assert "cough syrup" in ans
    assert "[MOCK LLM RESPONSE]" in ans

def test_llm_service_anti_hallucination():
    """Verify strict anti-hallucination compliance response when context is irrelevant or missing."""
    llm_service = LLMService()
    
    # Unrelated question to fever/cough contexts
    context = ["For fever, administer Paracetamol 500mg."]
    ans = llm_service.generate_answer("How to treat snake bite?", context, force_mock=True)
    assert ans == "I am sorry, but the provided clinical protocols do not contain information to answer this question."

    # Empty context
    ans = llm_service.generate_answer("What is the dose for malaria?", [], force_mock=True)
    assert ans == "I am sorry, but the provided clinical protocols do not contain information to answer this question."
