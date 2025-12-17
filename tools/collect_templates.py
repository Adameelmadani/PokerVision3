import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cv.capture import ScreenCapture

def collect_templates():
    print("Template Collection Tool")
    print("1. Capture new screenshot")
    print("2. Load existing screenshot")
    choice = input("Enter choice (1/2): ")
    
    img = None
    if choice == '1':
        cap = ScreenCapture()
        print("Capturing in 3 seconds...")
        cv2.waitKey(3000)
        img = cap.capture_screen()
        cv2.imwrite("data/screenshots/latest.png", img)
    else:
        path = input("Enter path to screenshot: ")
        img = cv2.imread(path)
        
    if img is None:
        print("Failed to load image.")
        return

    print("Select ROI and press SPACE or ENTER. Press c to cancel.")
    r = cv2.selectROI("Select Template", img)
    cv2.destroyWindow("Select Template")
    
    if r[2] == 0 or r[3] == 0:
        print("No region selected.")
        return
        
    # Crop
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    # Save
    name = input("Enter template name (e.g., Ah, dealer_btn): ")
    category = input("Category (cards/state): ")
    
    save_dir = f"data/templates/{category}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cv2.imwrite(f"{save_dir}/{name}.png", imCrop)
    print(f"Saved to {save_dir}/{name}.png")

if __name__ == "__main__":
    collect_templates()
