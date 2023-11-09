import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True)
observation, info = env.reset(seed=42)

for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print ("action : ", action, "\n")

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		observation, info = env.reset()

env.close()




