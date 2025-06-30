import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import os

# Load environment with rendering
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load trained model
model = PPO.load("models/ppo_cartpole")

# Prepare frame storage
frames = []
obs, _ = env.reset()

for step in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    frame = env.render()
    frames.append(frame)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Save video
os.makedirs("videos", exist_ok=True)
video_path = "videos/cartpole_simulation.mp4"
imageio.mimsave(video_path, frames, fps=30)

print(f"Video saved at {video_path}")
