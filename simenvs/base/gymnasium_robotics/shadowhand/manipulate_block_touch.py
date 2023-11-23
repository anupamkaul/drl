import gymnasium as gym

# code: https://github.com/Farama-Foundation/Gymnasium-Robotics
# doc : https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block_touch_sensors/

'''
Description
This environment was introduced in “Using Tactile Sensing to Improve the Sample Efficiency and Performance 
of Deep Deterministic Policy Gradients for Simulated In-Hand Manipulation Tasks”.

The environment is based on the same robot hand as in the HandReach environment, the Shadow Dexterous Hand. 
The task to be solved is the same as in the HandManipulateBlock environment. However, in this case the 
environment observation also includes tactile sensory information. This is achieved by placing a total of 
92 MuJoCo touch sensors in the palm and finger phalanxes of the hand. The sensors are created by specifying 
the tactile sensors’ active zones by so-called sites. Each site can be represented as either ellipsoid 
(for the finger tips) or box (for the phalanxes and palm sensors). When rendering the environment the sites 
are visualized as red and green transparent shapes attached to the hand model. If a body’s contact point 
falls within a site’s volume and involves a geometry attached to the same body as the site, the corresponding
contact force is included in the sensor reading. 

Soft contacts do not influence the above computation except inasmuch as the contact point might move outside 
of the site, in which case if a contact point falls outside the sensor zone, but the normal ray intersects 
the sensor zone, it is also included. MuJoCo touch sensors only report normal forces using Minkowski Portal 
Refinement approach . The output of the contact sensor is a non-negative scalar value of type float that is 
computed as the sum of all contact normal forces that were included for this sensor in the current time step. 
Thus, each sensor of the 92 virtual touch sensors has a non-negative scalar value.

The sensors are divided between the areas of the tip, middle, and lower phalanx of the forefinger, middle, ring, 
and little fingers In addition to the areas of the three thumb phalanxes and the paml. The number of sensors are 
divided as follows in the different defined areas of the hand: (see accompanying doc for this file
in docs folder or see https://robotics.farama.org/envs/shadow_dexterous_hand/manipulate_block_touch_sensors/)

When adding the sensors to the HandManipulateBlock environment there are two possible environment initializationsa
depending on the type of data returned by the touch sensors. This data can be continuous values of external forces 
or a boolean value which is True if the sensor detects any contact force and False if not. 

This two types of environments can be initialized from the environment id variations of HandManipulateBlock by adding 
the _ContinuousTouchSensors string to the id if the touch sensors return continuous force values or 
_BooleanTouchSensors if the values are boolean.

Continuous Touch Sensor Environments:
HandManipulateBlock_ContinuousTouchSensors-v1

HandManipulateBlockRotateZ_ContinuousTouchSensors-v1

HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1

HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1

HandManipulateBlockFull_ContinuousTouchSensors-v1

Boolean Touch Sensor Environments:
HandManipulateBlock_BooleanTouchSensors-v1

HandManipulateBlockRotateZ_BooleanTouchSensors-v1

HandManipulateBlockRotateParallel_BooleanTouchSensors-v1

HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1

HandManipulateBlockFull_BooleanTouchSensors-v1

The Action Space, Rewards, Starting State, Episode End, and Arguments sections are the same as for the HandManipulateBlock 
environment and its variations.

'''

import time #only to visually see end of episode

# continuous touch sensor environments
#env = gym.make("HandManipulateBlock_ContinuousTouchSensors-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateZ_ContinuousTouchSensors-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1", max_episode_steps=150, render_mode="human")
# doesn't exist: env = gym.make("HandManipulateBlockFull_ContinuousTouchSensors-v1", max_episode_steps=150, render_mode="human")

# boolean touch sensor environments (boolean values only)
#env = gym.make("HandManipulateBlock_BooleanTouchSensors-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateZ_BooleanTouchSensors-v1", max_episode_steps=150, render_mode="human")
#env = gym.make("HandManipulateBlockRotateParallel_BooleanTouchSensors-v1", max_episode_steps=150, render_mode="human")
env = gym.make("HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1", max_episode_steps=150, render_mode="human")

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
time.sleep(2)

# loop to observe states and send back random actions

total_reward=0
observation, info = env.reset(seed=42)


for _ in range(1000):

    action = env.action_space.sample()
    print(_, " action: ", action, "\n")
    time.sleep(0.5) # slow the simulation just to visually observe

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

