import numpy as np
import gymnasium as gym
from src.rl.poker_env import PokerEnv
from src.cv.state_builder import StateBuilder

def verify_alignment():
    print("Verifying State Alignment...")
    
    # 1. Initialize Env and StateBuilder
    env = PokerEnv(num_players=6)
    builder = StateBuilder(initial_stack=100.0)
    
    # 2. Create a Mock CV State
    mock_cv_state = {
        "hand": ["Ah", "Kd"],
        "board": ["2s", "3d", "4c"],
        "pot": 15.5,
        "players": [
            {"id": 0, "status": "active", "stack": 90.0, "bet": 10.0},
            {"id": 1, "status": "folded", "stack": 95.0, "bet": 5.0},
            {"id": 2, "status": "active", "stack": 100.0, "bet": 0.0},
            {"id": 3, "status": "active", "stack": 100.0, "bet": 0.0},
            {"id": 4, "status": "active", "stack": 80.0, "bet": 20.0}, # Hero
            {"id": 5, "status": "active", "stack": 100.0, "bet": 0.0},
        ]
    }
    
    # 3. Build Observation
    obs = builder.build_observation(mock_cv_state)
    
    # 4. Check against Env Space
    print("Checking observation space compliance...")
    try:
        # We can't use env.observation_space.contains(obs) directly because 
        # Box space checks might be strict on float32 vs float64 or exact bounds.
        # But let's try.
        is_valid = env.observation_space.contains(obs)
        if is_valid:
            print("SUCCESS: Observation is valid within Gym Space.")
        else:
            print("WARNING: Observation might be out of bounds or wrong type.")
            # Debugging
            for key, val in obs.items():
                space = env.observation_space[key]
                if not space.contains(val):
                    print(f"Key '{key}' invalid.")
                    print(f"Value: {val}")
                    print(f"Space: {space}")
    except Exception as e:
        print(f"Error checking space: {e}")

    # 5. Print details
    print("\nGenerated Observation:")
    for k, v in obs.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    verify_alignment()
