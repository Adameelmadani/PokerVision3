import mss
import numpy as np
import cv2
import time

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()

    def capture_screen(self, monitor_number=1):
        """Captures the full screen of the specified monitor."""
        monitor = self.sct.monitors[monitor_number]
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        # Convert BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def capture_region(self, region):
        """
        Captures a specific region.
        region: dict with 'top', 'left', 'width', 'height'
        """
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def save_screenshot(self, filename="screenshot.png"):
        """Captures and saves the full screen."""
        img = self.capture_screen()
        cv2.imwrite(filename, img)
        print(f"Screenshot saved to {filename}")

if __name__ == "__main__":
    # Test capture
    cap = ScreenCapture()
    cap.save_screenshot("test_capture.png")
