import os
import time
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from utils.logger import logger
from rag.rag_service import RAGService
from llm.llm_service import LLMService
from asr.asr_service import ASRService
from translation.translation_service import TranslationService
from vision.vision_service import VisionService
from tts.tts_service import TTSService
from config.config import settings

router = APIRouter(prefix="/assistant", tags=["Clinical Assistant"])

# Instantiate services once at the module level to reuse models (singleton pattern)
asr_service = ASRService()
translation_service = TranslationService()
tts_service = TTSService()
rag_service = RAGService()
llm_service = LLMService()
vision_service = VisionService()

def detect_indic_language(text: str) -> str:
    """Detects Indian languages based on Unicode character ranges, falling back to English."""
    if not text:
        return "en"
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            return "hi"  # Devnagari (Hindi, Marathi, etc. - default to 'hi')
        elif 0x0980 <= cp <= 0x09FF:
            return "bn"  # Bengali / Assamese
        elif 0x0A00 <= cp <= 0x0A7F:
            return "pa"  # Gurmukhi (Punjabi)
        elif 0x0A80 <= cp <= 0x0AFF:
            return "gu"  # Gujarati
        elif 0x0B00 <= cp <= 0x0B7F:
            return "or"  # Oriya
        elif 0x0B80 <= cp <= 0x0BFF:
            return "ta"  # Tamil
        elif 0x0C00 <= cp <= 0x0C7F:
            return "te"  # Telugu
        elif 0x0C80 <= cp <= 0x0CFF:
            return "kn"  # Kannada
        elif 0x0D00 <= cp <= 0x0D7F:
            return "ml"  # Malayalam
    return "en"

@router.post("/interact")
async def interact(
    audio_file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    language: str = Form("en")
):
    """
    Main interactive clinical pipeline endpoint.
    Processes voice or text input, processes symptom image if present,
    runs RAG protocols search, prompts LLM, translates response, and generates TTS.
    """
    logger.info(f"Interaction received. Lang: {language}, Has Audio: {audio_file is not None}, Has Image: {image_file is not None}")
    
    # Resolve language dynamically using auto-detection if set to 'auto'
    resolved_lang = language
    if language == "auto" and text_input:
        resolved_lang = detect_indic_language(text_input)
        if resolved_lang == "en":
            # Check for transliterated text
            resolved_lang = translation_service.detect_transliterated_language(text_input)
        logger.info(f"Auto-detected language from text_input: '{resolved_lang}'")

    if not audio_file and not text_input:
        raise HTTPException(status_code=400, detail="Either audio_file or text_input must be provided.")

    # Process query text (ASR vs Direct Text Input)
    detected_text = ""
    if audio_file:
        temp_dir = settings.get_absolute_path("logs")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"upload_{int(time.time())}.wav")
        try:
            with open(temp_path, "wb") as f:
                f.write(await audio_file.read())
            logger.info(f"Saved uploaded audio file to {temp_path}")
            
            # Run ASR
            detected_text = asr_service.transcribe(temp_path, language=resolved_lang)
            logger.info(f"ASR Transcribed Text: '{detected_text}'")
            
            # Post-ASR language detection for audio queries
            if language == "auto":
                resolved_lang = detect_indic_language(detected_text)
                if resolved_lang == "en":
                    # Check for transliterated text
                    resolved_lang = translation_service.detect_transliterated_language(detected_text)
                logger.info(f"Auto-detected language from transcribed audio: '{resolved_lang}'")
        except Exception as e:
            logger.error(f"ASR pipeline file processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
        finally:
            # Cleanup temp WAV file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    else:
        detected_text = text_input

    logger.info(f"Checking if query translation is required for language: '{resolved_lang}'")
    try:
        # If language was auto-detected or is one of the local languages, and we have a running Ollama model,
        # we can attempt translation using the LLM to handle transliterated inputs perfectly.
        # Otherwise, fall back to the standard translation service.
        if language == "auto" and llm_service.is_ollama_running():
            detected_lang, translated_query = llm_service.translate_transliterated_query(detected_text)
            if detected_lang != "en":
                resolved_lang = detected_lang
                logger.info(f"Ollama transliterated detector resolved: lang={resolved_lang}, translated='{translated_query}'")
            else:
                translated_query = translation_service.translate_input_if_needed(detected_text, language=resolved_lang)
        else:
            translated_query = translation_service.translate_input_if_needed(detected_text, language=resolved_lang)
        logger.info(f"Resolved translated query: '{translated_query}'")
    except Exception as e:
        logger.error(f"Failed to translate input: {e}")
        translated_query = detected_text
    
    
    # Integrate Visual Analysis
    extracted_symptoms = ""
    if image_file:
        logger.info(f"Image upload received: {image_file.filename}")
        temp_dir = settings.get_absolute_path("logs")
        os.makedirs(temp_dir, exist_ok=True)
        temp_img_path = os.path.join(temp_dir, f"upload_img_{int(time.time())}.jpg")
        try:
            with open(temp_img_path, "wb") as f:
                f.write(await image_file.read())
            logger.info(f"Saved uploaded image to {temp_img_path}")
            
            extracted_symptoms = vision_service.extract_symptoms(temp_img_path)
            logger.info(f"Vision extracted symptoms: '{extracted_symptoms}'")
        except Exception as e:
            logger.error(f"Failed to process uploaded image: {e}")
            extracted_symptoms = "Error processing image."
        finally:
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                  # Capture exception
                except Exception:
                    pass

    # Integrate RAG Retrieval
    logger.info(f"Querying vector database for: '{translated_query}'")
    try:
        retrieved_chunks = rag_service.retrieve_context(translated_query, top_k=2)
        matched_protocols = [chunk["content"] for chunk in retrieved_chunks]
        if not matched_protocols:
            matched_protocols = ["No protocol files indexed yet. Please upload medical guidelines."]
    except Exception as e:
        logger.error(f"Failed to retrieve context from RAG: {e}")
        matched_protocols = [f"RAG search error: {str(e)}"]

    # Integrate LLM Generation
    logger.info("Generating answer from LLM...")
    try:
        response_en = llm_service.generate_answer(
            question=translated_query,
            context_chunks=matched_protocols,
            visual_symptoms=extracted_symptoms if extracted_symptoms else None
        )
    except Exception as e:
        logger.error(f"Failed to generate answer from LLM: {e}")
        response_en = f"LLM generation error: {str(e)}"
    
    # Translate response automatically when required
    logger.info(f"Checking if response translation is required for language: '{resolved_lang}'")
    try:
        response_local = translation_service.translate_output_if_needed(response_en, language=resolved_lang)
        logger.info("Successfully resolved local response translation.")
    except Exception as e:
        logger.error(f"Failed to translate output response: {e}")
        response_local = response_en
    
    # Integrate TTS Synthesis
    logger.info("Generating response audio via TTS...")
    audio_filename = "latest_speech.wav"
    audio_dir = settings.get_absolute_path(settings.TTS_AUDIO_OUTPUT_DIR)
    os.makedirs(audio_dir, exist_ok=True)
    audio_output_path = os.path.join(audio_dir, audio_filename)
    
    try:
        tts_success = tts_service.synthesize(
            text=response_local,
            language=resolved_lang,
            output_path=audio_output_path
        )
        audio_url = f"/audio/{audio_filename}?t={int(time.time())}" if tts_success else None
    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}")
        audio_url = None
        audio_output_path = None

    return {
        "success": True,
        "input_type": "audio" if audio_file else "text",
        "detected_text": detected_text,
        "translated_query": translated_query,
        "extracted_symptoms": extracted_symptoms,
        "matched_protocols": matched_protocols,
        "response_en": response_en,
        "response_local": response_local,
        "audio_response_path": audio_output_path,
        "audio_response_url": audio_url
    }
