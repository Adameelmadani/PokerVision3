import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from poker_env import PokerEnv

def main():
    # Create environment
    env = PokerEnv(num_players=6)
    
    # Check if the environment follows Gym interface
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Initialize PPO agent
    # We use MultiInputPolicy because observation is a Dict
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train the agent
    print("Training agent...")
    model.learn(total_timesteps=10000)
    
    # Save the agent
    save_path = "ppo_poker_agent"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Test the agent
    obs, _ = env.reset()
    for _ in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
