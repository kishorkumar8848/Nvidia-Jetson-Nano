import os
import sys
import time
import threading
import requests
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.io.wavfile as wav

# Try to import audio libraries
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Backend endpoints
BACKEND_URL = "http://127.0.0.1:8000/api/assistant/interact"
HEALTH_URL = "http://127.0.0.1:8000/api/status"

class DesktopMockUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Suno Sutra Handheld UI - Jetson Display Mock")
        self.root.geometry("640x520")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        # State variables
        self.asr_lang = "auto"  # Default Auto-Detect
        self.recording = False
        self.audio_thread = None
        self.temp_filename = "logs/ui_input.wav"
        self.captured_image_path = None
        os.makedirs("logs", exist_ok=True)

        self.setup_styles()
        self.create_widgets()
        self.check_server_status()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1E1E1E")
        style.configure("TopBar.TFrame", background="#0A0A0A")
        
        # Configure fonts
        self.font_title = ("Helvetica", 14, "bold")
        self.font_text = ("Helvetica", 12)
        self.font_status = ("Helvetica", 10, "italic")
        
    def create_widgets(self):
        # 1. Top Bar Frame
        self.topbar = ttk.Frame(self.root, style="TopBar.TFrame", height=40)
        self.topbar.pack(fill=tk.X, side=tk.TOP)
        self.topbar.pack_propagate(False)

        # Mode Indicator
        self.mode_label = tk.Label(
            self.topbar, text="MODE: Auto-Detect (auto)", bg="#0A0A0A", fg="#777777",
            font=("Helvetica", 10, "bold")
        )
        self.mode_label.pack(side=tk.LEFT, padx=15)

        # Settings Toggle Button (mimics handheld book icon)
        self.settings_btn = tk.Button(
            self.topbar, text="Settings ⚙", bg="#333333", fg="#FFFFFF",
            activebackground="#555555", activeforeground="#FFFFFF", bd=0, padx=10,
            command=self.toggle_settings
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # Status indicators (Battery & Memory mock)
        self.stats_label = tk.Label(
            self.topbar, text="RAM: 42% | BAT: 98% 🔋", bg="#0A0A0A", fg="#FFFFFF",
            font=("Helvetica", 10)
        )
        self.stats_label.pack(side=tk.RIGHT, padx=15)

        # 2. Main Screen Area (320x240 scaled ratio)
        self.screen_frame = tk.Frame(self.root, bg="#121212", highlightbackground="#333333", highlightthickness=2)
        self.screen_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Top Text Box (ASR Output / Input query)
        self.lbl_top_title = tk.Label(self.screen_frame, text="Patient Input / ASR Transcript", bg="#121212", fg="#777777", font=("Helvetica", 9, "bold"))
        self.lbl_top_title.pack(anchor=tk.W, padx=10, pady=(5, 0))
        
        self.toptext = tk.Text(
            self.screen_frame, height=5, bg="#181818", fg="#00FF66", insertbackground="white",
            font=self.font_text, bd=0, highlightthickness=1, highlightbackground="#222222", padx=10, pady=5
        )
        self.toptext.pack(fill=tk.X, padx=10, pady=5)
        self.toptext.insert(tk.END, "Press 'RECORD' or type query and click 'SEND'...")

        # Bottom Text Box (TTS Input / Translated Response)
        self.lbl_bottom_title = tk.Label(self.screen_frame, text="Clinical Assistant Output", bg="#121212", fg="#777777", font=("Helvetica", 9, "bold"))
        self.lbl_bottom_title.pack(anchor=tk.W, padx=10, pady=(5, 0))
        
        self.bottomtext = tk.Text(
            self.screen_frame, height=7, bg="#181818", fg="#33CCFF", insertbackground="white",
            font=self.font_text, bd=0, highlightthickness=1, highlightbackground="#222222", padx=10, pady=5
        )
        self.bottomtext.pack(fill=tk.X, padx=10, pady=5)
        self.bottomtext.insert(tk.END, "")

        # 3. Control Panel (Record & Send buttons)
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=(0, 10))

        # Record Button
        self.record_btn = tk.Button(
            self.controls_frame, text="🔴 RECORD (5s)", bg="#990000", fg="#FFFFFF",
            activebackground="#CC0000", activeforeground="#FFFFFF", font=self.font_title,
            bd=0, height=2, width=15, command=self.start_record_thread
        )
        self.record_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # Send Text Button
        self.send_btn = tk.Button(
            self.controls_frame, text="✉ SEND QUERY", bg="#0066CC", fg="#FFFFFF",
            activebackground="#0080FF", activeforeground="#FFFFFF", font=self.font_title,
            bd=0, height=2, width=15, command=self.send_text_query
        )
        self.send_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Camera Button
        self.camera_btn = tk.Button(
            self.controls_frame, text="📷 CAMERA", bg="#228B22", fg="#FFFFFF",
            activebackground="#2E8B57", activeforeground="#FFFFFF", font=self.font_title,
            bd=0, height=2, width=12, command=self.trigger_camera_capture
        )
        self.camera_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # Clear Screen Button
        self.clear_btn = tk.Button(
            self.controls_frame, text="🗑 CLEAR", bg="#444444", fg="#FFFFFF",
            activebackground="#666666", activeforeground="#FFFFFF", font=self.font_title,
            bd=0, height=2, width=8, command=self.clear_screen
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # 4. Status Bar
        self.statusbar = tk.Label(
            self.root, text="Initializing...", bg="#0A0A0A", fg="#777777",
            font=self.font_status, anchor=tk.W, padx=10, pady=4
        )
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        # 5. Settings Pop-up Frame (Hidden by default)
        self.settings_frame = tk.Frame(
            self.screen_frame, bg="#1E1E1E", highlightbackground="#444444", highlightthickness=2
        )
        # Place it centrally over the screen area
        self.settings_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.8, relheight=0.8)
        
        # Settings widgets
        lbl_sett = tk.Label(self.settings_frame, text="Configure Language Settings", bg="#1E1E1E", fg="#FFFFFF", font=self.font_title)
        lbl_sett.pack(pady=10)

        # Language Select Radio Buttons
        self.lang_var = tk.StringVar(value="auto")
        
        lbl_lang = tk.Label(self.settings_frame, text="ASR & Response Language:", bg="#1E1E1E", fg="#CCCCCC", font=self.font_text)
        lbl_lang.pack(pady=5)
        
        frame_radio = tk.Frame(self.settings_frame, bg="#1E1E1E")
        frame_radio.pack(pady=5)
        
        tk.Radiobutton(
            frame_radio, text="Auto-Detect (auto)", variable=self.lang_var, value="auto",
            bg="#1E1E1E", fg="#FFFFFF", selectcolor="#1E1E1E", activebackground="#1E1E1E", activeforeground="#FFFFFF",
            command=self.save_lang_settings
        ).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(
            frame_radio, text="Hindi (hi)", variable=self.lang_var, value="hi",
            bg="#1E1E1E", fg="#FFFFFF", selectcolor="#1E1E1E", activebackground="#1E1E1E", activeforeground="#FFFFFF",
            command=self.save_lang_settings
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(
            frame_radio, text="Tamil (ta)", variable=self.lang_var, value="ta",
            bg="#1E1E1E", fg="#FFFFFF", selectcolor="#1E1E1E", activebackground="#1E1E1E", activeforeground="#FFFFFF",
            command=self.save_lang_settings
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(
            frame_radio, text="Malayalam (ml)", variable=self.lang_var, value="ml",
            bg="#1E1E1E", fg="#FFFFFF", selectcolor="#1E1E1E", activebackground="#1E1E1E", activeforeground="#FFFFFF",
            command=self.save_lang_settings
        ).pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(
            frame_radio, text="English (en)", variable=self.lang_var, value="en",
            bg="#1E1E1E", fg="#FFFFFF", selectcolor="#1E1E1E", activebackground="#1E1E1E", activeforeground="#FFFFFF",
            command=self.save_lang_settings
        ).pack(side=tk.LEFT, padx=10)

        # Close button
        btn_close = tk.Button(
            self.settings_frame, text="Save & Close", bg="#0066CC", fg="#FFFFFF",
            bd=0, padx=15, pady=5, command=self.toggle_settings
        )
        btn_close.pack(pady=20)
        
        # Hide settings initially
        self.settings_visible = False
        self.settings_frame.place_forget()

    def check_server_status(self):
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                self.statusbar.config(text="Status: Connected to FastAPI backend server", fg="#00FF66")
            else:
                self.statusbar.config(text="Status: Backend returned error", fg="#FF3333")
        except Exception:
            self.statusbar.config(text="Status: Offline (fastapi server not running at :8000)", fg="#FF3333")

    def toggle_settings(self):
        if self.settings_visible:
            self.settings_frame.place_forget()
            self.settings_visible = False
        else:
            self.settings_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.8, relheight=0.8)
            self.settings_visible = True

    def save_lang_settings(self):
        self.asr_lang = self.lang_var.get()
        if self.asr_lang == "hi":
            lang_name = "Hindi"
        elif self.asr_lang == "ta":
            lang_name = "Tamil"
        elif self.asr_lang == "ml":
            lang_name = "Malayalam"
        elif self.asr_lang == "auto":
            lang_name = "Auto-Detect"
        else:
            lang_name = "English"
        self.mode_label.config(text=f"MODE: {lang_name} ({self.asr_lang})")

    def clear_screen(self):
        self.toptext.delete("1.0", tk.END)
        self.bottomtext.delete("1.0", tk.END)
        self.captured_image_path = None
        self.statusbar.config(text="Cleared screen and active image.")

    def start_record_thread(self):
        if self.recording:
            return
        if not HAS_AUDIO:
            messagebox.showerror("Error", "sounddevice or soundfile not installed. Recording is disabled.")
            return

        self.recording = True
        self.record_btn.config(text="Recording...", bg="#333333", state=tk.DISABLED)
        self.statusbar.config(text="Recording audio inputs from mic...", fg="#33CCFF")
        
        self.audio_thread = threading.Thread(target=self.record_and_process)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def record_and_process(self):
        try:
            sample_rate = 16000
            duration = 5
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            wav.write(self.temp_filename, sample_rate, recording)
            
            # Send file to server
            self.root.after(0, self.send_audio_query, self.temp_filename)
        except Exception as e:
            self.root.after(0, self.show_error, f"Recording failed: {str(e)}")

    def send_audio_query(self, audio_path):
        self.statusbar.config(text="Transcribing and running clinical pipeline...", fg="#FFFF33")
        self.record_btn.config(text="🔴 RECORD (5s)", bg="#990000", state=tk.NORMAL)
        self.recording = False

        def process_request():
            try:
                files = {'audio_file': (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/wav')}
                if self.captured_image_path and os.path.exists(self.captured_image_path):
                    files['image_file'] = (os.path.basename(self.captured_image_path), open(self.captured_image_path, 'rb'), 'image/jpeg')

                data = {"language": self.asr_lang}
                
                response = requests.post(BACKEND_URL, data=data, files=files, timeout=60)
                files['audio_file'][1].close()
                if 'image_file' in files:
                    files['image_file'][1].close()
                
                if response.status_code == 200:
                    res = response.json()
                    # Clear image cache after query
                    self.captured_image_path = None
                    self.root.after(0, self.update_ui_results, res)
                else:
                    self.root.after(0, self.show_error, f"Error {response.status_code}: {response.text}")
            except Exception as e:
                self.root.after(0, self.show_error, f"Pipeline call failed: {str(e)}")

        threading.Thread(target=process_request, daemon=True).start()

    def send_text_query(self):
        text = self.toptext.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please type a clinical query first.")
            return

        self.statusbar.config(text="Processing text query...", fg="#FFFF33")

        def process_request():
            try:
                data = {
                    "language": self.asr_lang,
                    "text_input": text
                }
                files = {}
                if self.captured_image_path and os.path.exists(self.captured_image_path):
                    files['image_file'] = (os.path.basename(self.captured_image_path), open(self.captured_image_path, 'rb'), 'image/jpeg')

                response = requests.post(BACKEND_URL, data=data, files=files if files else None, timeout=60)
                if 'image_file' in files:
                    files['image_file'][1].close()

                if response.status_code == 200:
                    res = response.json()
                    # Clear image cache after query
                    self.captured_image_path = None
                    self.root.after(0, self.update_ui_results, res)
                else:
                    self.root.after(0, self.show_error, f"Error {response.status_code}: {response.text}")
            except Exception as e:
                self.root.after(0, self.show_error, f"Pipeline call failed: {str(e)}")

        threading.Thread(target=process_request, daemon=True).start()

    def trigger_camera_capture(self):
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Error", "OpenCV (opencv-python) is not installed. Camera preview is disabled.")
            return

        self.statusbar.config(text="Opening camera preview...", fg="#FFFF33")
        
        # Initialize video capture (index 0 for primary camera)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.warning("No camera hardware found. Falling back to simulated clinical placeholder.")
            if self.cap:
                self.cap.release()
            self.cap = None
            
            # Generate placeholder directly
            from vision.camera_capture import CameraCapture
            img_path = "logs/ui_capture.jpg"
            success = CameraCapture.capture_frame(img_path)
            if success:
                self.captured_image_path = img_path
                self.statusbar.config(text="Simulated capture successful!", fg="#00FF66")
                self.toptext.delete("1.0", tk.END)
                self.toptext.insert(tk.END, "Simulated skin rash placeholder generated! Ask a question and click 'SEND QUERY'...")
                messagebox.showinfo("Camera Fallback", "No webcam detected. Generated a simulated skin rash image placeholder in logs/ui_capture.jpg.")
            else:
                self.show_error("Failed to generate simulated image.")
            return

        # Create a Toplevel window for live preview
        self.cam_window = tk.Toplevel(self.root)
        self.cam_window.title("Live Camera Capture")
        self.cam_window.geometry("420x380")
        self.cam_window.configure(bg="#1E1E1E")
        self.cam_window.resizable(False, False)
        
        # Center the pop-up over the main window
        self.cam_window.transient(self.root)
        self.cam_window.grab_set()

        # Canvas/Label to show the video frames
        self.cam_label = tk.Label(self.cam_window, bg="#000000")
        self.cam_label.pack(pady=10, fill=tk.BOTH, expand=True)

        # Snapshot Button
        self.snap_btn = tk.Button(
            self.cam_window, text="📸 SNAPSHOT / CAPTURE", bg="#228B22", fg="#FFFFFF",
            font=self.font_title, bd=0, height=2, command=self.take_snapshot
        )
        self.snap_btn.pack(fill=tk.X, side=tk.BOTTOM)

        # Close stream if user closes window using 'X'
        self.cam_window.protocol("WM_DELETE_WINDOW", self.close_camera_window)

        # Start updating frames
        self.update_camera_frame()

    def update_camera_frame(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return
        
        try:
            import cv2
            from PIL import Image, ImageTk
            
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Flip image horizontally for natural mirror effect
                cv_img = cv2.flip(cv_img, 1)
                
                # Resize image to fit in the preview label (e.g. 400x300)
                pil_img = Image.fromarray(cv_img)
                pil_img = pil_img.resize((400, 300), Image.Resampling.LANCZOS)
                
                # Convert to Tkinter PhotoImage
                self.tk_img = ImageTk.PhotoImage(image=pil_img)
                self.cam_label.config(image=self.tk_img)
        except Exception as e:
            logger.error(f"Error reading camera frame: {e}")

        # Schedule next frame update in 20 milliseconds (~50 FPS)
        if hasattr(self, 'cam_window') and self.cam_window.winfo_exists():
            self.cam_window.after(20, self.update_camera_frame)

    def take_snapshot(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        try:
            import cv2
            ret, frame = self.cap.read()
            if ret:
                # Save the frame to logs folder
                img_path = "logs/ui_capture.jpg"
                os.makedirs("logs", exist_ok=True)
                cv2.imwrite(img_path, frame)
                
                if os.path.exists(img_path):
                    self.captured_image_path = img_path
                    self.statusbar.config(text="Camera capture successful!", fg="#00FF66")
                    self.toptext.delete("1.0", tk.END)
                    self.toptext.insert(tk.END, "Camera frame captured! Describe symptoms or ask a question and click 'SEND QUERY'...")
                    messagebox.showinfo("Camera Success", "Successfully captured image.\n\nNow enter a text query or record audio, and press 'SEND QUERY'.")
                else:
                    messagebox.showerror("Error", "Failed to save snapshot file.")
            else:
                messagebox.showerror("Error", "Failed to read frame from camera device.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed taking snapshot: {str(e)}")
        finally:
            self.close_camera_window()

    def close_camera_window(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'cam_window') and self.cam_window.winfo_exists():
            self.cam_window.destroy()
        self.statusbar.config(text="Camera closed.")

    def update_ui_results(self, result):
        # Update Text Box Contents
        self.toptext.delete("1.0", tk.END)
        self.toptext.insert(tk.END, result.get("detected_text", ""))

        self.bottomtext.delete("1.0", tk.END)
        response_text = result.get("response_local", "")
        self.bottomtext.insert(tk.END, response_text)

        self.statusbar.config(text="Success: Pipeline execution completed", fg="#00FF66")

        # Play TTS Audio Response
        audio_path = result.get("audio_response_path")
        if audio_path and os.path.exists(audio_path):
            threading.Thread(target=self.play_audio_file, args=(audio_path,), daemon=True).start()

    def play_audio_file(self, filepath):
        if not HAS_AUDIO:
            return
        try:
            data, fs = sf.read(filepath, dtype='float32')
            sd.play(data, fs)
            sd.wait()
        except Exception:
            pass

    def show_error(self, message):
        self.record_btn.config(text="🔴 RECORD (5s)", bg="#990000", state=tk.NORMAL)
        self.recording = False
        self.statusbar.config(text=message, fg="#FF3333")
        messagebox.showerror("Pipeline Error", message)

def run_main():
    # Detect if we should use SPI interface (Jetson Nano with physical display)
    # or fallback to Desktop Mock UI (Tkinter)
    use_spi = False
    try:
        import board
        import digitalio
        # Check if fourwire and adafruit_ili9341 are available
        import fourwire
        import adafruit_ili9341
        use_spi = True
    except Exception:
        pass

    if use_spi:
        print("[*] SPI Display components detected. Initializing ILI9341 physical display...")
        # Since it requires specific board settings, we run the handheld loop
        from ui.handheld import ILI9341UIConfig, IlI9341HandheldUI
        # Instantiate with default pins (defined in suno-sutra repo)
        config = ILI9341UIConfig(
            reset_pin="GP36_SPI3_CLK", # Change as per specific hardware pin maps
            pwm_pin="18",
            cs_pin="8",
            dc_pin="22",
            touch_cs="7",
            touch_irq="25"
        )
        print("[*] SPI Display initialized. Running display touch loop...")
        # Run ILI9341 touch controller loop
        import fourwire
        import adafruit_ili9341
        import xpt2046_circuitpython as xpt2046
        import displayio
        
        displayio.release_displays()
        # Initialize display bus
        spi = board.SPI()
        display_bus = fourwire.FourWire(spi, command=board.D22, chip_select=board.D8, baudrate=24000000)
        display = adafruit_ili9341.ILI9341(display_bus, width=320, height=240, rotation=90)
        touch = xpt2046.Touch(spi, cs=digitalio.DigitalInOut(board.D7), interrupt=digitalio.DigitalInOut(board.D25), force_baudrate=1000000)
        
        from ui.handheld import HandheldUI
        UI = HandheldUI(display, touch)
        
        # Hook up setting buttons to FastAPI parameters
        # Loop touch
        while True:
            UI.check_touch()
            time.sleep(0.1)
    else:
        print("[*] No SPI display found. Falling back to local Desktop Mock GUI (Tkinter)...")
        root = tk.Tk()
        app = DesktopMockUI(root)
        root.mainloop()

if __name__ == '__main__':
    run_main()
