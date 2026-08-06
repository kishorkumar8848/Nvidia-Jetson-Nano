import requests
from config.config import settings
from utils.logger import logger
from llm.prompt_templates import format_prompt

class LLMService:
    """Manages connection to local Ollama service and generates protocol-based clinical answers."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL
        logger.info(f"LLMService initialized targeting Ollama: {self.base_url} with model: {self.model}")

    def is_ollama_running(self) -> bool:
        """Sends a GET health request to verify if local Ollama server is alive."""
        try:
            # Ollama root endpoint returns "Ollama is running"
            response = requests.get(self.base_url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def generate_answer(self, question: str, context_chunks: list[str], visual_symptoms: str = None, force_mock: bool = False) -> str:
        """
        Generates clinical protocol-based response by injecting context & visual symptoms into prompt and calling Ollama.
        Falls back to a descriptive mock response if Ollama is unreachable.
        """
        prompt = format_prompt(question, context_chunks, visual_symptoms)
        
        if force_mock or not self.is_ollama_running():
            logger.warning("Ollama service is unreachable or force_mock is active. Falling back to mock model response.")
            return self._generate_mock_response(question, context_chunks, visual_symptoms)
            
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,      # Deterministic answers, no creative hallucination
                "top_p": 0.9,
                "num_predict": 256       # Limit output length for faster execution on Jetson
            }
        }
        
        try:
            logger.info(f"Sending generation request to Ollama: Model={self.model}")
            response = requests.post(url, json=payload, timeout=30.0)
            if response.status_code == 200:
                result_json = response.json()
                answer = result_json.get("response", "").strip()
                logger.info("Successfully received answer from Ollama.")
                return answer
            else:
                logger.error(f"Ollama returned error code {response.status_code}: {response.text}")
                return self._generate_mock_response(question, context_chunks, visual_symptoms)
        except Exception as e:
            logger.error(f"Inference request to Ollama failed: {e}")
            return self._generate_mock_response(question, context_chunks, visual_symptoms)

    def _generate_mock_response(self, question: str, context_chunks: list[str], visual_symptoms: str = None) -> str:
        """Fallback mock responses for testing and offline integration validation."""
        # Setup context and text variables
        context_text = " ".join(context_chunks).lower() if context_chunks else ""
        question_text = question.lower()
        symptoms_text = visual_symptoms.lower() if visual_symptoms else ""
        
        # Check if duration/day is provided in the question
        has_duration = any(word in question_text for word in ["day", "days", "duration", "days,", "day,"])
        is_direct_question = any(word in question_text for word in ["should", "how", "what", "treat", "dose", "gargle", "medicine", "protocol"])

        # 1. Ear Pain
        if "ear" in question_text:
            if has_duration or is_direct_question:
                return "[MOCK LLM RESPONSE] Based on clinical guidelines: Keep the ear clean and dry. Avoid self-medication with drops. Refer to a doctor if pain is severe."
            else:
                return "How many days have you had the ear pain, how severe is it, and is it in one ear or both?"

        # 2. Fever / Temperature / Rash
        if "fever" in question_text or "temperature" in question_text or "rash" in symptoms_text:
            if "rash" in symptoms_text:
                return "[MOCK LLM RESPONSE] Based on clinical protocols: Patient has skin rash and fever. If rash or stiff neck is present, refer immediately. Otherwise administer Paracetamol."
            
            if has_duration or is_direct_question:
                if "paracetamol" in context_text:
                    return "[MOCK LLM RESPONSE] Based on the clinical protocols: For high fever, apply cold compress and administer Paracetamol 500mg."
                else:
                    return "[MOCK LLM RESPONSE] GENERAL CLINICAL ADVICE (Not in retrieved protocols): For fever, apply cold compress and keep patient hydrated."
            else:
                return "How many days have you had the fever, how high is it, and are there other symptoms like headache or body aches?"

        # 3. Cough / Cold
        if "cough" in question_text or "cold" in question_text or "runny nose" in question_text:
            if has_duration or is_direct_question:
                if "cough syrup" in context_text:
                    return "[MOCK LLM RESPONSE] Based on the clinical guidelines: Advise warm fluids, avoid cold water, and prescribe cough syrup."
                else:
                    return "[MOCK LLM RESPONSE] GENERAL CLINICAL ADVICE (Not in retrieved protocols): Advise warm saline water gargles and recommend steam inhalation."
            else:
                return "How many days have you had the cough or cold, is it dry or productive, and do you have any difficulty breathing?"

        # 4. Stomach Pain
        if "stomach" in question_text or "abdomen" in question_text or "abdominal" in question_text:
            if has_duration or is_direct_question:
                return "[MOCK LLM RESPONSE] GENERAL CLINICAL ADVICE (Not in retrieved protocols): Stomach pain can have many causes. Avoid heavy foods, keep hydrated, and refer to a doctor if pain is severe."
            else:
                return "How many days have you had the stomach pain, how severe is it, and are there other symptoms like vomiting or fever?"
                
        # General advice fallback for custom inputs, otherwise strict warning for test compatibility
        if "eye" in question_text or "pain" in question_text or "general" in question_text:
            if has_duration or is_direct_question:
                return f"[MOCK LLM RESPONSE] GENERAL CLINICAL ADVICE (Not in retrieved protocols): For general symptoms related to '{question}', recommend keeping the area clean, offering fluids, avoiding self-medication, and consulting a healthcare professional."
            else:
                # Extract the symptom name from the question (e.g. eye pain, back pain)
                symptom = "pain"
                for s in ["eye", "back", "joint", "throat", "chest", "muscle", "stomach", "body"]:
                    if s in question_text:
                        symptom = f"{s} pain"
                        break
                return f"How many days have you had the {symptom}, how severe is it, and are there other symptoms present?"
            
        return "I am sorry, but the provided clinical protocols do not contain information to answer this question."

    def translate_transliterated_query(self, text: str) -> tuple[str, str]:
        """
        Detects language and translates transliterated Indian language queries to English.
        Returns:
            (detected_lang, translated_text)
            detected_lang: 'ta', 'hi', 'ml', or 'en'
            translated_text: Translated query in English
        """
        if not self.is_ollama_running():
            logger.warning("Ollama is not running. Cannot translate transliterated query using LLM.")
            return "en", text
            
        prompt = (
            "You are a translator. Detect the language and translate the patient query below to clean clinical English. Do not add medical advice or diagnosis. Just translate the words literally.\n\n"
            "Examples:\n"
            "Query: \"Enakku kaathu valikuthu\"\n"
            "Language: ta\n"
            "Translation: My ear hurts\n\n"
            "Query: \"mujhe bukhar hai\"\n"
            "Language: hi\n"
            "Translation: I have a fever\n\n"
            "Query: \"enniku pani undu\"\n"
            "Language: ml\n"
            "Translation: I have a fever\n\n"
            "Query: \"I have a headache\"\n"
            "Language: en\n"
            "Translation: I have a headache\n\n"
            f"Query: \"{text}\"\n"
        )
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 128
            }
        }
        
        try:
            logger.info("Calling Ollama to detect and translate query...")
            response = requests.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                result_json = response.json()
                answer = result_json.get("response", "").strip()
                logger.info(f"Ollama detection response:\n{answer}")
                
                # Parse response
                lang = "en"
                translation = text
                for line in answer.split("\n"):
                    if line.lower().startswith("language:"):
                        lang = line.split(":", 1)[1].strip().lower()
                        lang = lang.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(".", "")
                    elif line.lower().startswith("translation:"):
                        translation = line.split(":", 1)[1].strip()
                        translation = translation.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                
                # Normalize language code
                if lang in ["tamil", "ta"]:
                    lang = "ta"
                elif lang in ["hindi", "hi"]:
                    lang = "hi"
                elif lang in ["malayalam", "ml"]:
                    lang = "ml"
                else:
                    lang = "en"
                    
                return lang, translation
        except Exception as e:
            logger.error(f"Failed to run translation through Ollama: {e}")
            
        return "en", text

