import gymnasium as gym

# https://gymnasium.farama.org/environments/mujoco/walker2d/

#env = gym.make('Swimmer-v2', render_mode="human") # needs mujoco-py
env = gym.make('Swimmer-v4', ctrl_cost_weight=0.1, render_mode="human") 

# v4 reports error in var "collision" (gymnasium/envs/mujoco/assets/swimmer.xml)
# the only way to get by it was to remove collision attribute in swimmer.xml (I kept a copy)

observation, info = env.reset(seed=42)

import time

for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print ("action : ", action, "\n")

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		time.sleep(1)
		observation, info = env.reset()

env.close()




