import os
from utils.logger import logger

class CameraCapture:
    """Manages OpenCV local hardware camera capture with pillow-based fallback generation."""

    @staticmethod
    def capture_frame(output_path: str, camera_index: int = 0) -> bool:
        """
        Attempts to capture a frame from the specified camera index and save it as JPEG.
        Falls back to generating a simulated JPEG if no hardware webcam is active.
        """
        logger.info(f"Initiating camera frame capture (index: {camera_index})...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            import cv2
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video device at index {camera_index}")

            # Capture frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Failed to capture image frame from video device.")

            # Save frame
            cv2.imwrite(output_path, frame)
            logger.info(f"Webcam capture saved successfully to: {output_path}")
            return True

        except Exception as e:
            logger.warning(
                f"Local camera hardware capture failed: {e}. "
                "Generating a simulated symptom image placeholder."
            )
            return CameraCapture._generate_placeholder_image(output_path)

    @staticmethod
    def _generate_placeholder_image(output_path: str) -> bool:
        """Generates a solid color JPEG image block to simulate a medical photo capture."""
        try:
            from PIL import Image, ImageDraw
            # Create a 300x300 red/pink block representing a simulated skin rash image
            img = Image.new("RGB", (300, 300), color=(255, 180, 180))
            draw = ImageDraw.Draw(img)
            # Draw a circle simulating skin inflammation
            draw.ellipse([80, 80, 220, 220], fill=(220, 80, 80), outline=(255, 0, 0))
            
            img.save(output_path, "JPEG")
            logger.info(f"Simulated placeholder image saved to {output_path}")
            return True
        except Exception as err:
            logger.error(f"Failed to generate simulated image placeholder: {err}")
            return False
