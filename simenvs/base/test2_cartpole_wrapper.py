import gymnasium as gym

# showing usage of wrapper to :

# explore vs exploit (assume sample() means sampling from a learnt policy/agent. Then at epsilon
# randomize the action or set it to be explcit (as currently the policy is stochastic). Show wrapper's
# usage in expore vs exploit). Choose ActionWrapper, ObservationWrapper, RewardWrapper

# show video recorder capabilities (how to use monitoring of episodes)

# explore other wrappers from Gymnasium (inherit some of the predefined wrappers and extend them
# to achieve additional functionality

env = gym.make("CartPole-v1", render_mode="human")
#env = gym.wrappers.Monitor(env, "recording") #deprecated

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
