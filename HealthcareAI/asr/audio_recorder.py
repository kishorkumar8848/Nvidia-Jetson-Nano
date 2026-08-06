import os
import numpy as np
import scipy.io.wavfile as wav
from utils.logger import logger

class AudioRecorder:
    """Manages audio capture from local microphone inputs, producing 16kHz mono WAV outputs."""
    
    @staticmethod
    def record_audio(output_path: str, duration: float = 3.0, sample_rate: int = 16000) -> bool:
        """
        Records mono audio from microphone for specified duration.
        Generates dummy/silent WAV if no audio input device is available.
        """
        logger.info(f"Preparing to record {duration}s of audio to {output_path}...")
        
        # Ensure parent folder exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            import sounddevice as sd
            # Record mono 16-bit PCM data
            logger.info("Recording started. Speak into the microphone...")
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()  # Block until recording finishes
            logger.info("Recording finished.")
            
            # Save audio
            wav.write(output_path, sample_rate, recording)
            logger.info(f"Audio file saved to {output_path}")
            return True
            
        except Exception as e:
            logger.warning(
                f"Audio recording hardware error: {e}. "
                "Generating silent placeholder WAV file."
            )
            # Create a silent wave data array (zeros)
            silent_data = np.zeros(int(duration * sample_rate), dtype=np.int16)
            try:
                wav.write(output_path, sample_rate, silent_data)
                logger.info(f"Silent placeholder WAV file saved to {output_path}")
                return True
            except Exception as file_err:
                logger.error(f"Failed to write silent WAV: {file_err}")
                return False
pre_recorded_audio = None
