# ASHA/ANM Handheld AI Clinical Assistant

An offline clinical assistant tailored for Nvidia Jetson Nano (with SPI display mock support) and local desktop execution. The system features multi-lingual speech transcription, clinical guideline RAG lookup, Ollama MedGemma reasoning, Indic script translation, and MoonDream-based visual symptom analysis.

---

## 🚀 Key Features

*   **Indic Language Speech (ASR) & Synthesis (TTS)**: Full support for major Indic languages including **Tamil (`ta`), Hindi (`hi`), Malayalam (`ml`), Telugu (`te`), Kannada (`kn`), Bengali (`bn`), Gujarati (`gu`), Marathi (`mr`), Punjabi (`pa`), and Oriya (`or`)**.
*   **Automatic Language Detection**: Built-in Unicode range-based auto-detector that resolves Indic scripts on-the-fly and translates output queries/responses to correct local languages.
*   **Offline LLM clinical reasoning**: Integrated with local Ollama service running `medgemma:4b` to provide structured medical answers.
*   **Medical Guideline RAG**: FAISS-based vector database index lookup to retrieve matching protocol guidelines for incoming symptoms.
*   **Live Camera Preview GUI**: Tkinter desktop UI with mirror-flip camera preview render from OpenCV. Let ANMs align the lens to patient symptoms, capture snapshots (`logs/ui_capture.jpg`), and send them directly to VLM analysis.
*   **Multimodal VLM Analysis**: Local MoonDream-VLM integration via Ollama to describe rashes, color, swelling, or skin irritations from photos.
*   **Audio logging controls**: Single-file playback strategy that logs the latest output speech to `logs/latest_speech.wav` and re-caches it dynamically to avoid cache locks.
*   **Hardware Display compatibility**: Dual-mode pocketinfer-styled display rendering. Boots with Adafruit DisplayIO SPI touch-drivers on Jetson Nano, and falls back to desktop Tkinter GUI simulating the screen panel on standard PCs.

---

## 🛠 Project Structure

```text
d:/translator/
├── HealthcareAI/             # Clinical assistant source code
│   ├── asr/                 # Audio transcription services
│   ├── backend/             # FastAPI REST endpoints & routes
│   ├── config/              # Environment config schemas
│   ├── data/protocols/      # Reference clinical guide text files
│   ├── database/            # SQLite historical logs DB
│   ├── llm/                 # MedGemma prompts and Ollama APIs
│   ├── logs/                # Session recordings, photos and speech outputs
│   ├── rag/                 # FAISS vector store index & loaders
│   ├── tests/               # Service verification modules
│   ├── translation/         # IndicTrans2 translator NMT wraps
│   ├── tts/                 # Voice synthesis engines
│   ├── ui/                  # DisplayIO font panels
│   ├── vision/              # Live camera captures & VLM service
│   ├── cli_test.py          # Command prompt interactive testing client
│   ├── run.py               # FastAPI clinical backend server boot
│   └── run_ui.py            # Local desktop / SPI Handheld GUI runner
│
├── bhashini_models/         # Local Bhashini checkpoint directory
│   ├── asr/                 # Conformer ONNX models
│   └── nmt/                 # CTranslate2 translation weights
│
└── .gitignore               # Config to prevent model and log pushes
```

---

## 📦 Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) (Pull the models: `ollama pull medgemma:4b` & `ollama pull moondream`)
*   Microsoft Visual C++ Build Tools (Required for compiling local packages)

### 2. Install dependencies
```powershell
cd d:/translator/HealthcareAI
pip install -r requirements.txt
pip install ollama opencv-python soundfile sounddevice gtts
```

### 3. Running the Backend Server
Start the clinical pipeline server on port `8000`:
```powershell
python run.py
```

### 4. Running the Clients
*   **Desktop/SPI GUI Mock**:
    ```powershell
    python run_ui.py
    ```
    Click **Settings** to toggle languages or choose **Auto-Detect**. Click **📷 CAMERA** to trigger the mirror-flip webcam preview, click **📸 SNAPSHOT** to freeze the frame, and ask a question!
*   **Command Prompt CLI Client**:
    ```powershell
    python cli_test.py
    ```

---

## 📝 Configuration (`.env`)
You can adjust the endpoints, models, and directory parameters inside the local `HealthcareAI/.env` file:
```env
LLM_MODEL=medgemma:4b
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL_PROVIDER=moondream
```
