import os
import time
import logging
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from inference.engine import Model, iso_to_flores


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nmt_inference")


# =========================================================
# CONSTANTS
# =========================================================
INDIC_LANGUAGES = set(iso_to_flores.keys())
REQUIRED_MODELS = {"en-indic", "indic-en", "indic-indic"}


# =========================================================
# INFERENCE CLASS
# =========================================================
class NMTInference:

    # -----------------------------------------------------
    # Init → Load all models once
    # -----------------------------------------------------
    def __init__(self, checkpoint_root="./checkpoints"):

        self.models = {}

        if not os.path.exists(checkpoint_root):
            raise RuntimeError(
                f"Checkpoint folder not found: {checkpoint_root}"
            )

        for folder in os.listdir(checkpoint_root):

            if folder not in REQUIRED_MODELS:
                continue

            model_path = os.path.join(
                checkpoint_root,
                folder,
                "ct2_int8_model"
            )

            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Missing model path: {model_path}"
                )

            # Dynamic device detection for CTranslate2
            try:
                import ctranslate2
                import sys
                # Force CPU on Windows to prevent cublas DLL errors during local testing,
                # while allowing CUDA GPU acceleration on Jetson Nano / Linux.
                if sys.platform == "win32":
                    use_device = "cpu"
                else:
                    use_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                use_device = "cpu"

            logger.info(f"Loading NMT model: {folder} on device: {use_device}")

            self.models[folder] = Model(
                model_path,
                device=use_device,
                input_lang_code_format="iso",
                model_type="ctranslate2"
            )

        if not self.models:
            raise RuntimeError("No NMT models loaded")

        logger.info("All NMT models loaded successfully")

    # -----------------------------------------------------
    # Translation Routing
    # -----------------------------------------------------
    def _route_translate(self, text: str, src: str, tgt: str) -> str:

        if src in INDIC_LANGUAGES and tgt in INDIC_LANGUAGES:

            return self.models["indic-indic"] \
                .paragraphs_batch_translate__multilingual(
                    [[text, src, tgt]]
                )[0]

        elif src in INDIC_LANGUAGES and tgt == "EN":

            return self.models["indic-en"] \
                .paragraphs_batch_translate__multilingual(
                    [[text, src, "en"]]
                )[0]

        elif src == "EN" and tgt in INDIC_LANGUAGES:

            return self.models["en-indic"] \
                .paragraphs_batch_translate__multilingual(
                    [[text, "en", tgt]]
                )[0]

        else:
            raise ValueError(
                f"Unsupported translation direction {src} → {tgt}"
            )

    # -----------------------------------------------------
    # Public Inference Method
    # -----------------------------------------------------
    def infer(self, text: str, src_lang: str, tgt_lang: str):

        start_time = time.time()

        # -------------------------
        # Validation
        # -------------------------
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Text too long")

        # -------------------------
        # Translation
        # -------------------------
        translated = self._route_translate(
            text,
            src_lang,
            tgt_lang
        )

        logger.info(
            f"NMT {src_lang}→{tgt_lang} | "
            f"Time: {time.time() - start_time:.3f}s"
        )

        return {
            "translated_text": translated,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "processing_time_sec": round(
                time.time() - start_time, 3
            )
        }
