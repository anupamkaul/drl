import gymnasium as gym

# https://gymnasium.farama.org/environments/mujoco/walker2d/

env = gym.make("Walker2d", render_mode="human")
observation, info = env.reset(seed=42)

import time

for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print ("action : ", action, "\n")

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		time.sleep(2)
		observation, info = env.reset()

env.close()




