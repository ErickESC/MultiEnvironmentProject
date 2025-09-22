import gym
import retro
import torch
from stable_baselines3 import PPO

# Create the Retro environment (e.g., Sonic the Hedgehog)
env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

# Wrap the environment if needed (e.g., for compatibility)
env = gym.wrappers.FrameStack(env, 4)

# Instantiate the PPO agent
model = PPO('CnnPolicy', env, verbose=1, device='auto')

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_sonic")

# Close the environment
env.close()