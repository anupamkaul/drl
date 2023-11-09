import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
#env = gym.wrappers.Monitor(env, "recording")

total_reward=0
observation, info = env.reset(seed=42)

import time #only to visually see end of episode

for _ in range(1000):

    action = env.action_space.sample()
    print("action: ", action, "\n")

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print("observation: ", observation, "\n reward: ", reward, "total reward: ", total_reward, "info: ", info, "\n")
	

    if terminated or truncated:
        print("end of episode\n")
        #input() -- this makes render window go BG
        time.sleep(2)
        total_reward=0
        observation, info = env.reset()

env.close()
