import gymnasium as gym

# code: https://github.com/Farama-Foundation/Gymnasium-Robotics

# doc : https://robotics.farama.org/envs/adroit_hand/
# doc : https://robotics.farama.org/envs/adroit_hand/adroit_door/
# doc : https://robotics.farama.org/envs/adroit_hand/adroit_hammer/
# doc : https://robotics.farama.org/envs/adroit_hand/adroit_pen/
# doc:  https://robotics.farama.org/envs/adroit_hand/adroit_relocate/


'''
Description

Adroit Hand
This environments consists of a Shadow Dexterous Hand attached to a free arm. 
The system can have up to 30 actuated degrees of freedom. 
There are 4 possible environments that can be initialized depending on the task to be solved:

AdroitHandDoor-v1: The hand has to open a door with a latch.
AdroitHandHammer-v1: The hand has to hammer a nail inside a board.
AdroitHandPen-v1: The hand has to manipulate a pen until it achieves a desired goal position and rotation.
AdroitHandRelocate-v1: The hand has to pick up a ball and move it to a target location.

A sparse reward variant of each environment is also provided. 
These environments have a reward of 10.0 for achieving the target goal, and -0.1 otherwise. They can be initialized via:

AdroitHandDoorSparse-v1
AdroitHandHammerSparse-v1
AdroitHandPenSparse-v1
AdroitHandRelocateSparse-v1

'''

env = gym.make("AdroitHandDoor-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandHammer-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandPen-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandRelocate-v1", max_episode_steps=150, render_mode="human")

#env = gym.make("AdroitHandDoorSparse-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandHammerSparse-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandPenSparse-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("AdroitHandRelocateSparse-v1", max_episode_steps=150, render_mode="human")


env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# The following always has to hold: (commented out because these attributes seem to be missing : to be checked)
#assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
#assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
#assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
#substitute_goal = obs["achieved_goal"].copy()
#substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
#substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
#substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)

# loop to observe states and send back random actions

total_reward=0
observation, info = env.reset(seed=42)

import time #only to visually see end of episode

for _ in range(1000):

    action = env.action_space.sample()
    print(_, " action: ", action, "\n")
    #time.sleep(0.2) # slow the simulation just to visually observe

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print("observation: ", observation, "\n reward: ", reward, "total reward: ", total_reward, "info: ", info, "\n")

    # note that observation values contain actual_goal and desired_goal vectors
	

    if terminated or truncated:
        print("end of episode\n")
        #input() -- this makes render window go BG
        time.sleep(2)
        total_reward=0
        observation, info = env.reset()

env.close()

