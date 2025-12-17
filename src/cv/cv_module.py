import cv2
import numpy as np
import time
from .capture import ScreenCapture
from .detection import CardDetector, StateDetector

class ComputerVision:
    def __init__(self, monitor_number=1):
        self.capture = ScreenCapture()
        self.card_detector = CardDetector()
        self.state_detector = StateDetector()
        self.monitor_number = monitor_number
        
        # Define regions (x, y, w, h)
        # These need to be calibrated by the user or auto-detected.
        # For now, we'll use placeholders or a config dict.
        self.regions = {
            "table": {"top": 0, "left": 0, "width": 1920, "height": 1080}, # Full screen default
            "community_cards": [
                {"top": 400, "left": 600, "width": 50, "height": 70}, # Flop 1
                {"top": 400, "left": 660, "width": 50, "height": 70}, # Flop 2
                {"top": 400, "left": 720, "width": 50, "height": 70}, # Flop 3
                {"top": 400, "left": 780, "width": 50, "height": 70}, # Turn
                {"top": 400, "left": 840, "width": 50, "height": 70}, # River
            ],
            "my_hand": [
                {"top": 600, "left": 900, "width": 50, "height": 70}, # Card 1
                {"top": 600, "left": 960, "width": 50, "height": 70}, # Card 2
            ],
            "pot": {"top": 350, "left": 800, "width": 100, "height": 40},
            "seats": [
                # 6 seats, need coordinates
                {"top": 100, "left": 100, "width": 200, "height": 150}, # Seat 0
                {"top": 100, "left": 800, "width": 200, "height": 150}, # Seat 1
                {"top": 100, "left": 1500, "width": 200, "height": 150}, # Seat 2
                {"top": 800, "left": 1500, "width": 200, "height": 150}, # Seat 3
                {"top": 800, "left": 800, "width": 200, "height": 150}, # Seat 4 (Hero?)
                {"top": 800, "left": 100, "width": 200, "height": 150}, # Seat 5
            ]
        }

    def get_state(self):
        """
        Captures screen and returns the raw CV state.
        """
        # 1. Capture Full Table
        # For speed, we might capture specific regions, but full screen is easier to sync.
        full_img = self.capture.capture_screen(self.monitor_number)
        
        state = {
            "hand": [],
            "board": [],
            "pot": 0.0,
            "players": []
        }
        
        # 2. Detect My Hand
        for region in self.regions["my_hand"]:
            crop = self._crop(full_img, region)
            card = self.card_detector.match_card(crop)
            state["hand"].append(card if card else "NoCard")
            
        # 3. Detect Board
        for region in self.regions["community_cards"]:
            crop = self._crop(full_img, region)
            card = self.card_detector.match_card(crop)
            state["board"].append(card if card else "NoCard")
            
        # 4. Detect Pot
        pot_crop = self._crop(full_img, self.regions["pot"])
        state["pot"] = self.state_detector.get_number_from_region(pot_crop)
        
        # 5. Detect Players
        for i, seat_region in enumerate(self.regions["seats"]):
            seat_crop = self._crop(full_img, seat_region)
            
            # Status
            status = self.state_detector.get_seat_status(seat_crop)
            
            # Stack (Need a sub-region for stack within seat region, simplified here)
            # Assuming stack is in bottom half of seat region
            h, w = seat_crop.shape[:2]
            stack_crop = seat_crop[h//2:, :] 
            stack = self.state_detector.get_number_from_region(stack_crop)
            
            # Bet (Need sub-region)
            bet = 0.0 # Placeholder
            
            state["players"].append({
                "id": i,
                "status": status,
                "stack": stack,
                "bet": bet
            })
            
        return state

    def _crop(self, img, region):
        t, l, w, h = region["top"], region["left"], region["width"], region["height"]
        return img[t:t+h, l:l+w]

if __name__ == "__main__":
    cv = ComputerVision()
    print(cv.get_state())
