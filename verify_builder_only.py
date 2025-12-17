import numpy as np
from src.cv.state_builder import StateBuilder

def verify_builder():
    print("Verifying StateBuilder (Isolated)...")
    builder = StateBuilder(initial_stack=100.0)
    
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
    
    obs = builder.build_observation(mock_cv_state)
    print("Observation built successfully.")
    print(obs)

if __name__ == "__main__":
    verify_builder()
