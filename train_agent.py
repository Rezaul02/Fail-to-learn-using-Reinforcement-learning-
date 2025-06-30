import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create training environment
env = gym.make("CartPole-v1")

# Create model
model = PPO("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=20000)

# Save model to models/ directory
os.makedirs("models", exist_ok=True)
model.save("models/ppo_cartpole")

env.close()
print("Training completed and model saved.")
