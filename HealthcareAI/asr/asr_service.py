import os
import sys
import base64
import numpy as np
import scipy.io.wavfile as wav
from config.config import settings
from utils.logger import logger

import importlib.util

# Add the bhashini_models folder to sys.path
BHASHINI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bhashini_models"))
ASR_INFER_PATH = os.path.join(BHASHINI_ROOT, "asr", "infer.py")

HAS_BHASHINI_ASR = False
ASRInference = None

if os.path.exists(ASR_INFER_PATH):
    try:
        spec = importlib.util.spec_from_file_location("bhashini_asr", ASR_INFER_PATH)
        bhashini_asr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bhashini_asr)
        ASRInference = bhashini_asr.ASRInference
        HAS_BHASHINI_ASR = True
        logger.info("Successfully loaded Bhashini ASRInference dynamically.")
    except Exception as e:
        logger.warning(f"Could not load ASRInference via importlib: {e}. ASR will use mock fallback.")
else:
    logger.warning(f"Bhashini ASR inference script not found at {ASR_INFER_PATH}. ASR will use mock fallback.")

class ASRService:
    """ASR Speech-to-Text service wrapper supporting Bhashini conformer models & mock fallbacks."""

    def __init__(self):
        self.model_provider = settings.ASR_MODEL_PROVIDER
        logger.info(f"ASRService initialized using provider: {self.model_provider}")
        
        self.asr_engine = None
        if HAS_BHASHINI_ASR:
            try:
                checkpoint_dir = os.path.join(BHASHINI_ROOT, "asr", "checkpoints")
                logger.info(f"Loading Bhashini ASR conformer models from {checkpoint_dir}")
                self.asr_engine = ASRInference(checkpoint_dir=checkpoint_dir)
            except Exception as e:
                logger.error(f"Failed to load Bhashini ASR models: {e}")

    def transcribe(self, audio_file_path: str, language: str = "en", force_mock: bool = False) -> str:
        """
        Transcribes a local WAV file to text using Bhashini models.
        Gracefully falls back to mock translation strings if checkpoints or engines are missing/unsupported.
        """
        logger.info(f"Transcribing audio file: {audio_file_path} (Language: {language})")
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return ""

        lang = language.lower()
        lang_code = "en"
        if lang in ["tamil", "ta"]:
            lang_code = "ta"
        elif lang in ["hindi", "hi"]:
            lang_code = "hi"
        elif lang in ["malayalam", "ml"]:
            lang_code = "ml"
        elif lang in ["telugu", "te"]:
            lang_code = "te"
        elif lang in ["kannada", "kn"]:
            lang_code = "kn"
        elif lang in ["bengali", "bn"]:
            lang_code = "bn"
        elif lang in ["gujarati", "gu"]:
            lang_code = "gu"
        elif lang in ["marathi", "mr"]:
            lang_code = "mr"
        elif lang in ["punjabi", "pa"]:
            lang_code = "pa"
        elif lang in ["oriya", "odia", "or"]:
            lang_code = "or"

        # Try high-quality SpeechRecognition (Google Web API) for dynamic transcription
        # especially for English (which has no local Bhashini conformer) and general fallback.
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio_data = r.record(source)
            
            # Map target language for speech recognizer
            g_lang = "en-US"
            if lang_code == "ta":
                g_lang = "ta-IN"
            elif lang_code == "hi":
                g_lang = "hi-IN"
            elif lang_code == "ml":
                g_lang = "ml-IN"
            elif lang_code == "te":
                g_lang = "te-IN"
            elif lang_code == "kn":
                g_lang = "kn-IN"
            elif lang_code == "bn":
                g_lang = "bn-IN"
            elif lang_code == "gu":
                g_lang = "gu-IN"
            elif lang_code == "mr":
                g_lang = "mr-IN"
            elif lang_code == "pa":
                g_lang = "pa-IN"
            elif lang_code == "or":
                g_lang = "or-IN"

            logger.info(f"Attempting SpeechRecognition (Google API) for language: {g_lang}")
            text = r.recognize_google(audio_data, language=g_lang)
            logger.info(f"SpeechRecognition resolved transcript: '{text}'")
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"SpeechRecognition fallback failed: {e}. Falling back to Bhashini/Mock.")

        if force_mock or lang_code == "en" or not self.asr_engine:
            if not self.asr_engine and lang_code != "en":
                logger.warning("Bhashini ASR engine is not loaded. Using mock fallback.")
            return self._generate_mock_transcript(audio_file_path, language)

        try:
            with open(audio_file_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            
            result = self.asr_engine.infer(audio_base64=audio_b64, language=lang_code)
            text = result.get("text", "")
            logger.info(f"Bhashini ASR transcript resolved: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Bhashini ASR inference failed: {e}. Falling back to mock.")
            return self._generate_mock_transcript(audio_file_path, language)

    def _generate_mock_transcript(self, audio_file_path: str, language: str) -> str:
        """Generates static mock transcription text mapped to language constraints."""
        logger.info("Generating mock ASR transcription...")
        
        try:
            sample_rate, data = wav.read(audio_file_path)
            is_silent = np.all(data == 0)
            logger.debug(f"Read WAV file with sample_rate={sample_rate}, elements={len(data)}, is_silent={is_silent}")
        except Exception as e:
            logger.warning(f"Could not parse WAV data structure: {e}")

        lang = language.lower()
        if lang in ["tamil", "ta"]:
            return "எனக்கு காய்ச்சல் மற்றும் தலைவலி உள்ளது"  # "I have a fever and headache"
        elif lang in ["hindi", "hi"]:
            return "मुझे बुखार और सिरदर्द है"             # "I have a fever and headache"
        elif lang in ["malayalam", "ml"]:
            return "എനിക്ക് പനിയും തലവേദനയും ഉണ്ട്"       # "I have a fever and headache" in Malayalam
        else:
            return "Patient has fever and headache."
pre_recorded_audio = None

