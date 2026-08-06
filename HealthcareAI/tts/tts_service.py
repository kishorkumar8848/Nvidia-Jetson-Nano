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
TTS_INFER_PATH = os.path.join(BHASHINI_ROOT, "tts", "infer.py")

HAS_BHASHINI_TTS = False
TTSInference = None

if os.path.exists(TTS_INFER_PATH):
    try:
        spec = importlib.util.spec_from_file_location("bhashini_tts", TTS_INFER_PATH)
        bhashini_tts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bhashini_tts)
        TTSInference = bhashini_tts.TTSInference
        HAS_BHASHINI_TTS = True
        logger.info("Successfully loaded Bhashini TTSInference dynamically.")
    except Exception as e:
        logger.warning(f"Could not load Bhashini TTSInference via importlib: {e}. TTS will use pyttsx3 fallback.")
else:
    logger.warning(f"Bhashini TTS inference script not found at {TTS_INFER_PATH}. TTS will use pyttsx3 fallback.")

class TTSService:
    """ASHA Text-to-Speech service converting clinical responses to offline speech audio using Bhashini / pyttsx3."""

    def __init__(self):
        self.provider = settings.TTS_MODEL_PROVIDER
        logger.info(f"TTSService initialized with provider: {self.provider}")
        
        self.bhashini_tts = None
        if HAS_BHASHINI_TTS:
            try:
                logger.info("Initializing Bhashini TTS Inference engine...")
                self.bhashini_tts = TTSInference()
            except Exception as e:
                logger.warning(f"Could not initialize Bhashini TTS engine: {e}. Fallback to pyttsx3 will be used.")

    def synthesize(self, text: str, language: str, output_path: str, force_mock: bool = False) -> bool:
        """
        Synthesizes localized text into offline voice WAV files using Bhashini TTS.
        Falls back to pyttsx3 or a simulator tone if Bhashini is offline/unsupported.
        """
        # Clean markdown symbols (like *, -, #, etc.) so TTS doesn't pronounce them (e.g. "star")
        import re
        clean_text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        clean_text = clean_text.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
        clean_text = re.sub(r'#+\s+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        logger.info(f"Synthesizing text to speech: '{clean_text[:40]}...' (Language: {language})")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if force_mock:
            logger.info("Force mock enabled. Generating mock speech file.")
            return self._write_mock_speech_wav(output_path)

        # 1. Attempt Bhashini TTS
        if self.bhashini_tts:
            try:
                # Map language code
                lang = language.lower()
                if lang in ["tamil", "ta"]:
                    lang_code = "ta"
                elif lang in ["hindi", "hi"]:
                    lang_code = "hi"
                elif lang in ["malayalam", "ml"]:
                    lang_code = "ml"
                else:
                    lang_code = "en"

                logger.info(f"Running Bhashini TTS engine for language: {lang_code}")
                result = self.bhashini_tts.infer(
                    text=clean_text,
                    language=lang_code,
                    return_base64=True
                )
                
                audio_bytes = base64.b64decode(result["audio_base64"])
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully synthesized Bhashini audio: {output_path}")
                    return True
                else:
                    raise RuntimeError("Bhashini output file size is empty.")
            except Exception as e:
                logger.warning(f"Bhashini TTS engine synthesis failed: {e}. Falling back to pyttsx3.")

        # 2. Try gTTS (Google Text-To-Speech) for high-quality translation testing on PC
        try:
            from gtts import gTTS
            import soundfile as sf
            import io

            lang = language.lower()
            if lang in ["tamil", "ta"]:
                gtts_lang = "ta"
            elif lang in ["hindi", "hi"]:
                gtts_lang = "hi"
            elif lang in ["malayalam", "ml"]:
                gtts_lang = "ml"
            elif lang in ["telugu", "te"]:
                gtts_lang = "te"
            elif lang in ["kannada", "kn"]:
                gtts_lang = "kn"
            elif lang in ["bengali", "bn"]:
                gtts_lang = "bn"
            elif lang in ["gujarati", "gu"]:
                gtts_lang = "gu"
            elif lang in ["marathi", "mr"]:
                gtts_lang = "mr"
            elif lang in ["punjabi", "pa"]:
                gtts_lang = "pa"
            elif lang in ["oriya", "odia", "or"]:
                gtts_lang = "or"
            else:
                gtts_lang = "en"

            logger.info(f"Attempting gTTS synthesis for language: {gtts_lang}")
            tts = gTTS(clean_text, lang=gtts_lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            # Decode the MP3 stream and write out as a standard Microsoft WAV file
            data, samplerate = sf.read(fp)
            sf.write(output_path, data, samplerate, format='WAV')

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully synthesized audio via gTTS: {output_path}")
                return True
        except Exception as e:
            logger.warning(f"gTTS synthesis failed: {e}. Trying offline pyttsx3 fallback...")

        # 3. Fallback to pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)

            voices = engine.getProperty('voices')
            lang = language.lower()
            
            matched_voice_id = None
            for voice in voices:
                voice_name = voice.name.lower()
                if lang in ["tamil", "ta"] and ("tamil" in voice_name or "kalpana" in voice_name):
                    matched_voice_id = voice.id
                    break
                elif lang in ["hindi", "hi"] and ("hindi" in voice_name or "hemant" in voice_name or "kalpana" in voice_name):
                    matched_voice_id = voice.id
                    break
                elif lang in ["malayalam", "ml"] and ("malayalam" in voice_name or "anjana" in voice_name or "kalpana" in voice_name):
                    matched_voice_id = voice.id
                    break

            if matched_voice_id:
                logger.info(f"Setting fallback TTS voice to: {matched_voice_id}")
                engine.setProperty('voice', matched_voice_id)
            else:
                logger.debug("No specific local script voice found. Falling back to default system voice.")

            engine.save_to_file(clean_text, output_path)
            engine.runAndWait()
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully synthesized fallback pyttsx3 audio: {output_path}")
                return True
            else:
                raise RuntimeError("pyttsx3 engine run complete but output file size is empty.")

        except Exception as e:
            logger.warning(
                f"Offline pyttsx3 synthesis failed: {e}. "
                "Generating simulated medical alert beep audio file."
            )
            return self._write_mock_speech_wav(output_path)

    def _write_mock_speech_wav(self, output_path: str, duration: float = 1.0, sample_rate: int = 16000) -> bool:
        """Writes a simple standard sine wave beep WAV to simulate synthesised voice audio."""
        try:
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 440 * t)
            audio_data = (tone * 16384).astype(np.int16)
            
            wav.write(output_path, sample_rate, audio_data)
            logger.info(f"Simulated audio fallback written to {output_path}")
            return True
        except Exception as err:
            logger.error(f"Failed to generate simulated WAV fallback: {err}")
            return False
pre_recorded_audio = None
