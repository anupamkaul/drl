import gymnasium as gym

# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

# -- play with environments (look at source of env in gymnasium to learn more)

env = gym.make("FrozenLake", render_mode="human") # default map_name = 4x4, is_slippery=True
#env = gym.make("FrozenLake", map_name="8x8", render_mode="human") # double the map size to 8 by 8
#env = gym.make("FrozenLake", map_name="8x8", is_slippery=False, render_mode="human") # non-slippery mode
#env = gym.make("FrozenLake", desc = None , map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

#from gymnasium.envs.toy_text.frozen_lake import generate_random_map
#env = gym.make("FrozenLake", desc = generate_random_map(8), map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

observation, info = env.reset(seed=42)

import time

def translate(action: int) -> str:
	if action==0:
		 return "LEFT"
	elif action==1:
		 return "DOWN"
	elif action==2:
		 return "RIGHT"
	elif action==3:
		 return "UP"


for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print (_, " next action : ", action, " (", translate(action), ")\n")
	time.sleep(1) # add a 1 second delay in steps to observe the next action...

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")
	time.sleep(1) # add a 1 second delay in steps to observe the observation...

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		#input()
		time.sleep(2)
		observation, info = env.reset()

env.close()




