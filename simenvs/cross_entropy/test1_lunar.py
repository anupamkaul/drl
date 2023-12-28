import gymnasium as gym
import torch.nn as nn

'''
Background
----------

During the agent's lifetime, its experience is presented as episodes. Every episode is
a sequence of observations that the agent has got from the environment, actions it has 
issued, and rewards for these actions. 

Imagine that our agent has played several episodes. For each episode we can calculate the
total reward agent has claimed (discounted or not-discounted). For simplicity lets assume
discount-factor of 1, i.e. just a sum of local rewards for every episode played. (The total
reward is a measure of goodness of that episode w.r.t the agent.

The core of cross-entropy method is to throw away bad episodes and train on better ones.
Steps are as follows:

1. Play N number of episodes using our current model and environment
2. Calculate the total reward for every episode and decide on a reward
   boundary. Usually we use some percentile of all rewards (e.g. 50th / 70th)
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observations as the input
   and issued actions as the desired output.
5. Repeat from step 1 till we are satisfied with the result.

Execution:
---------

Our model's (agent's) core is a one-hidden-layer neural network, with ReLU and
128 hidden neurons (this is arbitrary). Other hyperparameters are also set almost
randomly, as the method is robust and converges very quickly. 

'''
# https://gymnasium.farama.org/environments/classic_control/cart_pole/
env = gym.make("CartPole-v1", render_mode="human")
#env = gym.wrappers.Monitor(env, "recording")
observation, info = env.reset(seed=42)

HIDDEN_SIZE = 128 # single layer count of neurons
BATCH_SIZE  = 16  # count of episodes we play on every iteration``
PERCENTILE  = 70  # percentile of episodes' total rewards that we use for elite episode filtering 
                  # we take 70th percentile => we will leave top 30% of episodes sorted by reward

# this is the agent now:
class Net (nn.Module):
	def __init__(self, obs_size, hidden_size, n_actions):
		super(Net, self).__init__()
		self.net = nn.sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
	def forward(self, x):
		return self.net(x)



import time
for _ in range(1000):
	action = env.action_space.sample() # (take action) this is where I will insert my policy (sample for now)
	print (_, " action : ", action, "\n")

	observation, reward, terminated, truncated, info = env.step(action) # (observe the env-state, reward) 
	print ("observation : ", observation, "\nreward : ", reward, "\n")

	if terminated or truncated:
		print(" oops .. terminated or truncated !!\n")
		time.sleep(2)
		observation, info = env.reset()

env.close()




