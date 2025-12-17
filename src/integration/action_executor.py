import pyautogui
import time
import random

class ActionExecutor:
    def __init__(self):
        # Button coordinates (x, y)
        # These need to be calibrated by the user.
        self.buttons = {
            0: (1000, 900), # Fold
            1: (1150, 900), # Check/Call
            2: (1300, 900), # Raise
        }
        
        # Safety: Fail-safe corner
        pyautogui.FAILSAFE = True

    def execute_action(self, action_id):
        """
        Executes the given action ID (0=Fold, 1=Call, 2=Raise).
        """
        if action_id not in self.buttons:
            print(f"Unknown action: {action_id}")
            return

        x, y = self.buttons[action_id]
        
        # Add some human-like randomness
        x += random.randint(-10, 10)
        y += random.randint(-5, 5)
        
        print(f"Executing Action {action_id}: Clicking at ({x}, {y})")
        
        # Move and click
        # pyautogui.moveTo(x, y, duration=0.2)
        # pyautogui.click()
        
        # For safety during dev, we just print. Uncomment above to enable.
        print("DEBUG: Click simulated.")

if __name__ == "__main__":
    executor = ActionExecutor()
    executor.execute_action(1)
