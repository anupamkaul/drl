import gymnasium as gym

# code: https://github.com/Farama-Foundation/Gymnasium-Robotics
# doc : https://robotics.farama.org/envs/franka_kitchen/ and https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/

'''
Description

Multitask environment in which a 9-DoF Franka robot is placed in a kitchen containing several common household items. 
The goal of each task is to interact with the items in order to reach a desired goal configuration.

The tasks can be selected when the environment is initialized passing a list of tasks to the tasks_to_complete argument 
as follows:
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])

The possible tasks to complete are:
bottom burner : twist control knob to activate bottom left burner in the stove.
top burner    : twist control knob to activate top left burner in the stove.
light switch  : move a lever switch to turn on a light over the burners.
slide cabinet : slide open the cabinet door.
hinge cabinet : open a hinge cabinet door.
microwave     : open the microwave door.
kettle        : move the kettle from the bottom burner to the top burner.

Details:

This environment was introduced in “Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning” 
by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

The environment is based on the 9 degrees of freedom Franka robot. 
The Franka robot is placed in a kitchen environment containing several common household items: 
a microwave, a kettle, an overhead light, cabinets, and an oven. 

The environment is a multitask goal in which the robot has to interact with the previously mentioned items in order to 
reach a desired goal configuration. For example, one such state is to have the microwave and sliding cabinet door open with 
the kettle on the top burner and the overhead light on. The goal tasks can be configured when the environment is created.

Goal
The goal has a multitask configuration. 
The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argumenttasks_to_complete. 

For example, to open the microwave door and move the kettle create the environment as follows:
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])

Possible tasks with respective joint goal values, action space (joint actuators and joint velocities with mujoco), observation space,
info, rewards, starting state, episode end, arguements etc are described in documentation (pdf) and links above.

'''

import time #only to visually see end of episode

#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['bottom burner'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['top burner'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['light switch'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['slide cabinet', 'kettle'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['hinge cabinet', 'kettle'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode="human")
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'], render_mode="human")
#env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle', 'top burner', 'light switch'], render_mode="human")

env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# The following always has to hold:
assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
#assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
# assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
substitute_goal = obs["achieved_goal"].copy()
substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
#substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
#substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)
time.sleep(2)

# loop to observe states and send back random actions

total_reward=0
observation, info = env.reset(seed=42)


for _ in range(1000):

    action = env.action_space.sample()
    print(_, " action: ", action, "\n")
    # time.sleep(0.5) # slow the simulation just to visually observe

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

