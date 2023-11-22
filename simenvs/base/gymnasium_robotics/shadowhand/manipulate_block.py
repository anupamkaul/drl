import gymnasium as gym

# code: https://github.com/Farama-Foundation/Gymnasium-Robotics
# doc : https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block/

'''
Description
This environment was introduced in “Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research”.

The environment is based on the same robot hand as in the HandReach environment, the Shadow Dexterous Hand. 
In this task a block is placed on the palm of the hand. 
The task is to then manipulate the block such that a target pose is achieved. 
The goal is 7-dimensional and includes the target position (in Cartesian coordinates) and target rotation (in quaternions). 

In addition, variations of this environment can be used with increasing levels of difficulty:

HandManipulateBlockRotateZ-v1: Random target rotation around the z axis of the block. No target position.
HandManipulateBlockRotateParallel-v1: Random target rotation around the z axis of the block and axis-aligned target rotations for the x and y axes. No target position.
HandManipulateBlockRotateXYZ-v1: Random target rotation for all axes of the block. No target position.
HandManipulateBlockFull-v1: Random target rotation for all axes of the block. Random target position.

'''

#env = gym.make("HandManipulateBlock-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateZ-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateParallel-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateXYZ-v1", max_episode_steps=150, render_mode="human")
env = gym.make("HandManipulateBlockFull-v1", max_episode_steps=150, render_mode="human")

env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# The following always has to hold:
assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
substitute_goal = obs["achieved_goal"].copy()
substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)

# loop to observe states and send back random actions

total_reward=0
observation, info = env.reset(seed=42)

import time #only to visually see end of episode

for _ in range(1000):

    action = env.action_space.sample()
    print(_, " action: ", action, "\n")
    time.sleep(0.2) # slow the simulation just to visually observe

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

