import pytest
from translation.translation_service import TranslationService

def test_translation_skipping():
    """Verify that English-to-English translation skips model call and return original text."""
    service = TranslationService()
    
    # English input query
    text = "cough and cold symptoms"
    trans_input = service.translate_input_if_needed(text, language="en")
    assert trans_input == text

    # English output response
    response = "Keep patient warm and hydrate."
    trans_output = service.translate_output_if_needed(response, language="en")
    assert trans_output == response

def test_translation_indic_to_english():
    """Verify Indic input queries get translated to English correctly in mock fallback mode."""
    service = TranslationService()
    service.nmt_engine = None
    
    # Tamil to English
    ta_query = "எனக்கு காய்ச்சல் மற்றும் தலைவலி உள்ளது"
    eng_query_ta = service.translate_input_if_needed(ta_query, language="ta")
    assert eng_query_ta == "fever and headache"

    # Hindi to English
    hi_query = "मुझे बुखार और सिरदर्द है"
    eng_query_hi = service.translate_input_if_needed(hi_query, language="hi")
    assert eng_query_hi == "fever and headache"

def test_translation_english_to_indic():
    """Verify English clinical responses get translated back to localized scripts in mock fallback mode."""
    service = TranslationService()
    service.nmt_engine = None
    
    eng_response = "Administer Paracetamol 500mg and check fever."

    # English to Tamil
    ta_response = service.translate_output_if_needed(eng_response, language="ta")
    assert "பாராசிட்டமால்" in ta_response
    assert "மருத்துவ நெறிமுறைகளின்படி" in ta_response

    # English to Hindi
    hi_response = service.translate_output_if_needed(eng_response, language="hi")
    assert "पैरासिटामोल" in hi_response
    assert "नैदानिक ​​​​प्रोटोकॉल" in hi_response
