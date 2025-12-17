import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from src.cv.cv_module import ComputerVision
from src.cv.state_builder import StateBuilder
from src.integration.action_executor import ActionExecutor

def main():
    print("Initializing PokerVision3 Bot...")
    
    # 1. Load Components
    cv = ComputerVision()
    builder = StateBuilder(initial_stack=100.0)
    executor = ActionExecutor()
    
    # 2. Load Agent
    model_path = "ppo_poker_agent"
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except:
        print("Model not found! Please run src/rl/train_agent.py first.")
        return

    print("Bot is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            # 3. Capture & Detect
            raw_state = cv.get_state()
            
            # Check if it's my turn?
            # CV should ideally tell us if "Hero" is active/turn.
            # For now, we run continuously or wait for a specific trigger.
            # Let's assume we act if we have cards and it looks like our turn (not implemented yet).
            
            # 4. Build Observation
            obs = builder.build_observation(raw_state)
            
            # 5. Predict Action
            action, _states = model.predict(obs, deterministic=True)
            
            # 6. Execute Action
            # Only execute if we are confident it's our turn.
            # For safety, we just print the recommendation for now.
            print(f"Recommended Action: {action}")
            
            # executor.execute_action(int(action))
            
            # Sleep to avoid spamming
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("Bot stopped.")

if __name__ == "__main__":
    main()
