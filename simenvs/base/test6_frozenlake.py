import gymnasium as gym

# https://gymnasium.farama.org/environments/mujoco/humanoid_standup/

env = gym.make("FrozenLake", render_mode="human")
#env = gym.make("Acrobot", render_mode="human")
#env = gym.make("MountainCar", render_mode="human")
#env = gym.make("Pendulum", render_mode="human")
#env = gym.make("CliffWalking", render_mode="human")
#env = gym.make("Pusher", render_mode="human")
#env = gym.make("Reacher", render_mode="human")
#env = gym.make("Walker2d", render_mode="human")
#env = gym.make("HalfCheetah", render_mode="human")
#env = gym.make("Hopper", render_mode="human")
#env = gym.make("Ant", render_mode="human")
#env = gym.make("BipedalWalker", render_mode="human")
#env = gym.make("CarRacing", render_mode="human")
#env = gym.make("Blackjack", render_mode="human")
#env = gym.make("Taxi", render_mode="human")


observation, info = env.reset(seed=42)

import time

for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print ("action : ", action, "\n")

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		#input()
		time.sleep(2)
		observation, info = env.reset()

env.close()




