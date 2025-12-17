import cv2
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
from src.cv.detection import CardDetector

def verify_detection():
    # 1. Load test capture
    img = cv2.imread("test_capture.png")
    if img is None:
        print("test_capture.png not found")
        return

    # 2. Create a dummy template (crop 50x50 from 100,100)
    template = img[100:150, 100:150]
    if not os.path.exists("data/templates/cards"):
        os.makedirs("data/templates/cards")
    cv2.imwrite("data/templates/cards/test_card.png", template)
    print("Created dummy template: data/templates/cards/test_card.png")

    # 3. Run detector
    detector = CardDetector()
    # Pass the region where the card is (plus some context)
    # Let's pass the whole image or a larger region around 100,100
    region = img[50:200, 50:200]
    
    print("Running matching...")
    match = detector.match_card(region)
    print(f"Match result: {match}")
    
    if match == "test_card":
        print("SUCCESS: Detected test_card")
    else:
        print("FAILURE: Did not detect test_card")

if __name__ == "__main__":
    verify_detection()
