import cv2
import numpy as np
import os

class CardDetector:
    def __init__(self, templates_dir="data/templates/cards"):
        self.templates_dir = templates_dir
        self.templates = self._load_templates()

    def _load_templates(self):
        templates = {}
        if not os.path.exists(self.templates_dir):
            return templates
            
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".png"):
                name = filename.split(".")[0]
                img = cv2.imread(os.path.join(self.templates_dir, filename), 0) # Load as grayscale
                templates[name] = img
        return templates

    def match_card(self, image_region, threshold=0.8):
        """
        Matches a card in the given image region.
        Returns (rank, suit) or None.
        """
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_val = -1
        
        for name, template in self.templates.items():
            # Resize template if needed or ensure region is larger
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
                
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_match = name
                
        if best_val > threshold:
            # Assuming template names like 'Ah', 'Ks', '2d'
            # Or separate rank/suit templates. 
            # For simplicity, let's assume full card templates for now or rank/suit separate.
            # If full card: 'Ah' -> ('A', 'h')
            if len(best_match) == 2:
                return best_match[0], best_match[1]
            return best_match
            
        return None

class StateDetector:
    def __init__(self, templates_dir="data/templates/state"):
        self.templates_dir = templates_dir
        self.templates = self._load_templates()
        
    def _load_templates(self):
        templates = {}
        if not os.path.exists(self.templates_dir):
            return templates
            
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".png"):
                name = filename.split(".")[0]
                img = cv2.imread(os.path.join(self.templates_dir, filename), 0)
                templates[name] = img
        return templates

    def find_template(self, image, template_name, threshold=0.8):
        if template_name not in self.templates:
            return None
            
        template = self.templates[template_name]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure image is larger than template
        if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
            return None
            
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > threshold:
            return max_loc # Top-left corner
        return None

    def get_number_from_region(self, image_region):
        """
        Extracts a number from a region using digit template matching.
        Assumes templates '0'...'9' are in self.templates.
        """
        if image_region is None or image_region.size == 0:
            return 0.0
            
        # This is a simplified version. A robust OCR would use Tesseract or a CNN.
        # For template matching digits:
        # 1. Find all occurrences of 0-9.
        # 2. Sort by x-coordinate.
        # 3. Construct string.
        
        found_digits = []
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        
        for i in range(10):
            digit_name = str(i)
            if digit_name not in self.templates:
                continue
                
            template = self.templates[digit_name]
            if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                continue
                
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.85
            locs = np.where(res >= threshold)
            
            for pt in zip(*locs[::-1]):
                # pt is (x, y)
                # Filter duplicates (close points)
                is_duplicate = False
                for existing in found_digits:
                    if abs(existing[0] - pt[0]) < 5 and existing[1] == i: # Same digit, close x
                        is_duplicate = True
                        break
                if not is_duplicate:
                    found_digits.append((pt[0], i))
                    
        # Sort by x coordinate
        found_digits.sort(key=lambda x: x[0])
        
        # Filter overlapping detections (e.g. same digit detected multiple times slightly offset)
        final_digits = []
        if found_digits:
            last_x = -100
            for x, digit in found_digits:
                if x - last_x > 5: # Min width between digits
                    final_digits.append(str(digit))
                    last_x = x
                    
        if not final_digits:
            return 0.0
            
        try:
            return float("".join(final_digits))
        except:
            return 0.0

    def get_seat_status(self, seat_region):
        """
        Determines status of a seat: 'empty', 'active', 'folded'.
        """
        # Check for 'empty' seat template
        if self.find_template(seat_region, 'seat_empty'):
            return 'empty'
            
        # Check for 'folded' card backs or specific fold icon
        if self.find_template(seat_region, 'folded_icon'):
            return 'folded'
            
        # If not empty and not folded, assume active (or check for avatar/cards)
        return 'active'
