import os
import sys
import time
import requests
import numpy as np
import scipy.io.wavfile as wav

# Try to import sounddevice for audio recording and playback
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO_IO = True
except ImportError:
    HAS_AUDIO_IO = False

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000/api/assistant/interact"
HEALTH_URL = "http://127.0.0.1:8000/api/status"

def check_backend_running():
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False

def record_audio(filename="input_temp.wav", duration=5, sample_rate=16000):
    if not HAS_AUDIO_IO:
        print("\n[!] Error: sounddevice or soundfile not installed. Cannot record audio.")
        return False
        
    print(f"\n[*] Recording audio for {duration} seconds...")
    print("[*] Speak now...")
    try:
        # Record audio in mono, 16kHz
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("[*] Recording complete. Saving to file...")
        wav.write(filename, sample_rate, recording)
        return True
    except Exception as e:
        print(f"[!] Error during recording: {e}")
        return False

def play_audio(filepath):
    if not HAS_AUDIO_IO:
        print("\n[!] Error: sounddevice not installed. Cannot play audio.")
        return
    if not os.path.exists(filepath):
        print(f"\n[!] Audio file not found at {filepath}")
        return
        
    print(f"\n[*] Playing audio response: {filepath}...")
    try:
        data, fs = sf.read(filepath, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"[!] Error playing audio: {e}")

def main():
    print("==================================================")
    print("      Healthcare AI Clinical Assistant CLI        ")
    print("==================================================")
    
    # 1. Check if backend server is running
    print("[*] Checking backend server status...")
    if not check_backend_running():
        print("[!] Warning: FastAPI backend is not running at http://127.0.0.1:8000.")
        print("[!] Please run 'python run.py' in a separate terminal before running this CLI.")
        choice = input("[?] Do you want to continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            sys.exit(1)
            
    # 2. Setup testing directories
    temp_dir = "logs"
    os.makedirs(temp_dir, exist_ok=True)
    input_wav = os.path.join(temp_dir, "cli_input.wav")
    
    is_clarifying_turn = False
    mode = '3'
    language = 'auto'

    # 3. Main Loop
    while True:
        if not is_clarifying_turn:
            print("\n--------------------------------------------------")
            print("Select Input Mode:")
            print("1. Record Audio (Microphone)")
            print("2. Use Existing WAV file")
            print("3. Text Input Only")
            print("4. Exit")
            
            mode = input("Enter option (1-4): ").strip()
            
            if mode == '4':
                print("Exiting CLI. Goodbye!")
                break
                
            language = "auto"
            print("\nSelect Language:")
            print("Type the language code or press Enter for default:")
            print("Options: auto (Auto-Detect), hi (Hindi), ta (Tamil), ml (Malayalam), te (Telugu), kn (Kannada), bn (Bengali), gu (Gujarati), mr (Marathi), pa (Punjabi), or (Oriya), as (Assamese), ur (Urdu), ne (Nepali), sa (Sanskrit), brx (Bodo), doi (Dogri), ks (Kashmiri), kok (Konkani), mai (Maithili), mni (Manipuri), sat (Santali), sd (Sindhi), en (English)")
            lang_input = input("Enter language code [default=auto]: ").strip().lower()
            if lang_input in ["auto", "hi", "ta", "ml", "te", "kn", "bn", "gu", "mr", "pa", "or", "as", "ur", "ne", "sa", "brx", "doi", "ks", "kok", "mai", "mni", "sat", "sd", "en"]:
                language = lang_input
        else:
            print("\n--------------------------------------------------")
            print(f"[*] Clarifying Question Active. Reusing Mode={mode}, Language={language}")
            
        files = {}
        data = {"language": language}
        
        if mode == '1':
            if is_clarifying_turn:
                duration = 5
            else:
                duration = input("Enter recording duration in seconds [default=5]: ").strip()
                duration = int(duration) if duration.isdigit() else 5
            
            if record_audio(input_wav, duration=duration):
                files = {'audio_file': (os.path.basename(input_wav), open(input_wav, 'rb'), 'audio/wav')}
            else:
                is_clarifying_turn = False
                continue
                
        elif mode == '2':
            path = input("Enter path to WAV file: ").strip()
            if not os.path.exists(path):
                print(f"[!] File not found: {path}")
                is_clarifying_turn = False
                continue
            files = {'audio_file': (os.path.basename(path), open(path, 'rb'), 'audio/wav')}
            
        elif mode == '3':
            text = input("Enter your answer/query: ").strip() if is_clarifying_turn else input("Enter your clinical query: ").strip()
            if not text:
                print("[!] Query cannot be empty.")
                is_clarifying_turn = False
                continue
            data["text_input"] = text
            
        else:
            print("[!] Invalid option.")
            is_clarifying_turn = False
            continue
            
        # Send Request to Backend
        print("\n[*] Sending request to clinical pipeline...")
        start_time = time.time()
        try:
            response = requests.post(BACKEND_URL, data=data, files=files, timeout=60)
            elapsed = time.time() - start_time
            
            # Close file handle if open
            if 'audio_file' in files:
                files['audio_file'][1].close()
                
            if response.status_code == 200:
                result = response.json()
                print(f"\n[+] Pipeline execution completed successfully in {elapsed:.2f}s:")
                print(f"    - Input Type: {result.get('input_type')}")
                print(f"    - Transcription: {result.get('detected_text')}")
                print(f"    - Translated Query: {result.get('translated_query')}")
                print(f"    - RAG Match: {result.get('matched_protocols')}")
                print(f"    - Clinical Answer (EN): {result.get('response_en')}")
                print(f"    - Clinical Answer (Local): {result.get('response_local')}")
                
                # Check for output audio file
                audio_path = result.get("audio_response_path")
                if audio_path and os.path.exists(audio_path):
                    play_audio(audio_path)
                else:
                    print("[!] No TTS audio response returned or file does not exist.")
                
                # Check if the response is a clarifying question
                response_local = result.get('response_local', '').strip()
                response_en = result.get('response_en', '').strip()
                if response_local.endswith('?') or response_en.endswith('?'):
                    is_clarifying_turn = True
                else:
                    is_clarifying_turn = False
            else:
                print(f"[!] Pipeline returned error status {response.status_code}: {response.text}")
                is_clarifying_turn = False
        except requests.exceptions.Timeout:
            print("[!] Request timed out. Ensure backend is running and responsive.")
            is_clarifying_turn = False
        except Exception as e:
            print(f"[!] Request failed: {e}")
            is_clarifying_turn = False

if __name__ == '__main__':
    main()
