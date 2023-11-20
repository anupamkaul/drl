import gymnasium as gym

# https://gymnasium.farama.org/environments/mujoco/humanoid_standup/

# -- play with environments (look at source of env in gymnasium to learn more)
#env = gym.make("FrozenLake", render_mode="human") # default map_name = 4x4, is_slippery=True
#env = gym.make("FrozenLake", map_name="8x8", render_mode="human") # double the map size to 8 by 8
#env = gym.make("FrozenLake", map_name="8x8", is_slippery=False, render_mode="human") # non-slippery mode
#env = gym.make("FrozenLake", desc = None , map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

#from gym.envs.toy_text.frozen_lake import generate_random_map
#env = gym.make("FrozenLake", desc = generate_random_map(8), map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

#env = gym.make("Acrobot", render_mode="human")

#env = gym.make("MountainCar", render_mode="human")

#env = gym.make("Pendulum", render_mode="human")

#env = gym.make("CliffWalking", render_mode="human")

#env = gym.make("Pusher", render_mode="human")

#env = gym.make("Reacher", render_mode="human")

env = gym.make("Walker2d", render_mode="human")

# https://gymnasium.farama.org/environments/mujoco/
#env = gym.make("HalfCheetah", render_mode="human")

#env = gym.make("Hopper", render_mode="human")

#env = gym.make("Ant", render_mode="human")

#env = gym.make("BipedalWalker", render_mode="human")

#env = gym.make("CarRacing", render_mode="human")

#env = gym.make("Blackjack", render_mode="human")

#env = gym.make("Taxi", render_mode="human")

# atari games: https://gymnasium.farama.org/environments/atari/
# or the prev: https://www.gymlibrary.dev/environments/atari/
# pong: /home/anupam/github_drl/Gymnasium/docs/environments/atari/pong.md

# own env: https://www.gymlibrary.dev/content/environment_creation/


observation, info = env.reset(seed=42)

import time

for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print (_, " next action : ", action, "\n")
	#time.sleep(1) # add a 1 second delay in steps to observe the next action...

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")
	#time.sleep(1) # add a 1 second delay in steps to observe the observation...

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		time.sleep(2)
		observation, info = env.reset()

env.close()




