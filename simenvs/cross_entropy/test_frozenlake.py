import gymnasium as gym
import torch.nn as nn
import numpy as np

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
3. Throw away all episodes with a reward below the boundary. (this is Key)
4. Train on the remaining "elite" episodes using observations as the input
   and issued actions as the desired output. (calculate cross entropy everytime)
5. Repeat from step 1 till we are satisfied with the result.

Execution:
---------

Our model's (agent's) core is a one-hidden-layer neural network, with ReLU and
128 hidden neurons (this is arbitrary). Other hyperparameters are also set almost
randomly, as the method is robust and converges very quickly. 

'''
# -- play with environments (look at source of env in gymnasium to learn more)
#https://gymnasium.farama.org/environments/toy_text/frozen_lake/

#env = gym.make("FrozenLake", render_mode="human") # default map_name = 4x4, is_slippery=True
#env = gym.make("FrozenLake", map_name="8x8", render_mode="human") # double the map size to 8 by 8
#env = gym.make("FrozenLake", map_name="8x8", is_slippery=False, render_mode="human") # non-slippery mode
#env = gym.make("FrozenLake", desc = None , map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

'''
Since we're borrowing from cartpole's cross entropy model and that was based on vectorized observations, and
because frozen-lake's observation is a single int (range 0-15), we will write a wrapper for observations.
We can apply the traditional "one-hot encoding" of discrete inputs, which means that input to our network
will have 16 float numbers and zero everywhere, except the index that we will encode. To minimize code changes,
we will use the ObservationWrapper class from Gymnasium and implement our DiscreteOneHotWrapper class. (Also
look at base/cartpole_wrapper.py for an action-wrapper example)
'''

class DiscreteOneHotWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super(DiscreteOneHotWrapper, self).__init__(env)
		assert isinstance(env.observation_space, gym.spaces.Discrete)
		self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

	def observation(self, observation):
		res = np.copy(self.observation_space.low)
		res[observation] = 1.0
		return res

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
#env = gym.make("FrozenLake", desc = generate_random_map(8), map_name="8x8",  is_slippery=False, render_mode="human") # generate a random map each time

env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", desc = generate_random_map(8), map_name="8x8",  is_slippery=False, render_mode="human")) # generate a random map each time

#env = gym.wrappers.Monitor(env, "recording")
# observation, info = env.reset(seed=42)

HIDDEN_SIZE = 128 # single layer count of neurons
BATCH_SIZE  = 100 # count of episodes we play on every iteration
PERCENTILE  = 70  # percentile of episodes' total rewards that we use for elite episode filtering 
                  # we take 70th percentile => we will leave top 30% of episodes sorted by reward
GAMMA = 0.9

# this is the agent now:
class Net (nn.Module):
	def __init__(self, obs_size, hidden_size, n_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
	def forward(self, x):

		#print("net debug: incoming is len: ", len(x), "data: ", x)
		return self.net(x)

'''
Above is a pretty simple network: it takes a single observation from the environment as an input vector
and outputs a number for every action that we can perform. The output from the network is a probability
distribution over actions. 

Instead of including the traditional softmax non-linearity after the last step, (which uses exponentiation) 
and then calculating cross-entropy loss (which uses logarithm of probabilities) we'll use the pytorch class 
nn.CrossEntropyLoss which combines softmax and cross-entropy in a single, more numerically stable expression. 
(we will do this outside of the Net class invocation)
Cross-Entropy loss is based on KL divergence

Now let's define Episode and EpisodeStep. These are two helper classes:

EpisodeStep: This will be used to represent one single step (action) that our agent (the ML now) made in
the episode. EpisodeStep stores the observation from the environment and what action the agent completed.
(We will use episode steps from "elite" episode steps as training data for our ML algorithm.

Episode: This is a single episode itself, stored as total undiscounted reward and a collection of EpisodeStep
'''

from collections import namedtuple

Episode = namedtuple('Episode', field_names=['reward', 'steps'])  # remember steps are EpisodeSteps
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) 

'''
Now let's define a function that generates batches with episodes
'''

def iterate_batches(env, net, batch_size):

	batch = []
	episode_reward = 0.0
	episode_steps = []
	obs, info = env.reset(seed=42)
	sm = nn.Softmax(dim=1)
	steps = 0

	'''
	The above function accepts the environment (the Env class instance from Gymnasium library), our neural network,
	and the count of episodes it should generate on every iteration. We have the notion of a list of Episode instances
	that we call a batch. This list of Episodes will be captured by the variable 'batch' above (which is a list of Episode
	instances). We also declare a reward counter for the current episode and its list of steps(the EpisodeStep objects). 

	Then we reset the env to obtain the first observation and create a softmax layer, which will be used to convert the
	network's output to a probability distribution (between 0 and 1) of actions. 

	That was the prep work, now let's start the environment loop:

	'''

	import torch
	import time

	while True:
		#obs_v = torch.FloatTensor([obs])
		obs_v = torch.FloatTensor(np.array([obs]))
		#print(obs_v)
		act_probs_v = sm(net(obs_v))     # get action from the action probability density from Net...
		act_probs = act_probs_v.data.numpy()[0] 
		#print ("iterate_batches", " action probability: ", act_probs, "\n")
	
		# explanation of above

		action = np.random.choice(len(act_probs), p = act_probs)
		#print ("iterate_batches", " action: ", action, "\n")

		next_obs, reward, is_done, truncated, info = env.step(action)
		#print ("iterate_batches", " observation ", next_obs, "reward ", reward)
		steps = steps + 1

		# write some more

		episode_reward += reward
		#print ("iterate_batches", " episode_reward: ", episode_reward, "\n")
		episode_steps.append(EpisodeStep(observation=obs, action=action))

		if is_done:

			print("\riterate_batches", "episode ", len(batch) + 1, " TERMINATED (after ", steps, "steps)", end = ' ', flush=True)
			#print("batch up this episode..\n")
			#time.sleep(2)

			batch.append(Episode(reward=episode_reward, steps = episode_steps))	
			episode_reward = 0.0
			episode_steps = []
			next_obs, next_info  = env.reset(seed=42)
			steps = 0

			#exit loop
			if len(batch) == batch_size:
				yield batch

				batch = []
				steps = 0

		obs = next_obs


def filter_batch(batch, percentile):

	#rewards = list(map(lambda s: s.reward, batch))
	discounted_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
	reward_bound = np.percentile(discounted_rewards, percentile)

	#reward_mean = float(np.mean(rewards)) (handled in main loop now)

	#print("filter_batch ", "rewards: ", rewards, "\n")
	#print("filter_batch ", "reward_bound: ", reward_bound, "(this is the ", PERCENTILE, "th percentile)\n")
	#print("filter_batch ", "reward_mean: ", reward_mean, "\n")

	#explanations galore!

	train_obs = []
	train_act = []
	elite_batch = []

	for example, discounted_reward in zip(batch, discounted_rewards):

		if discounted_reward < reward_bound:
			train_obs.extend(map(lambda step: step.observation, example.steps))
			train_act.extend(map(lambda step: step.action, example.steps))
			elite_batch.append(example)

	# vectorization in main loop now
	return elite_batch, train_obs, train_act, reward_bound

import torch
PATH="./model/frozenlake_crossentropy"

def save_model(net):
	# check for existence(net), corner cases
	print("\nsaving model")
	torch.save(net, PATH)

def load_model():
	# check for existence (PATH), corner cases
	return torch.load(PATH)

# save model upon a SIGINT (control-C)
import signal

'''
valid_signals = signal.valid_signals()
print("Number of valid signals ", len(valid_signals), "\n")
for i in valid_signals:
	print(i)
'''

# define my signal handler
def SignalHandler_SIGINT(SignalNumber, Frame):
	#print("\nSignal Handler: CTRL-C was pressed!!\n")
	#print ("save current model\n")
	save_model(net)
	exit()

# register my signal handler
signal.signal(signal.SIGINT,SignalHandler_SIGINT) 

# main glue (continues from top)

# frozen lake's observation is a single int 
# obs: value representing the player’s current position as 
# current_row * nrows + current_col (where both the row and col start at 0)

# so the following call fails: we need a vector for observations (1X16) to re-use this code
# https://gymnasium.farama.org/api/wrappers/observation_wrappers/#

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# this creates Net of fixed size but obs size is different
# not because the number of observations are different (they are always 4)
# but because the sum total/flattened tensor then is different

net = Net(obs_size, HIDDEN_SIZE, n_actions)
net = load_model()

objective = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(params=net.parameters(), lr = 0.01)

from tensorboardX import SummaryWriter
writer = SummaryWriter()

#explanations

import time

'''
We call iterate_batches and filter them out, and then run our network,
optimize it, calculate our losses, and get our mean_rewards calculated
as many times until we're satisfied that mean_rewards is now a good number
to indicate the RL problem has been solved
'''

full_batch = []
for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

	#print("main: from iterate_batches ", "iter_no: ", iter_no, "batch: ", batch, "\n")
	#time.sleep(2)

	reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
	print("mean reward: ", reward_mean, "\n")

	#obs_v, acts_v, reward_b, reward_m = filter_batch(full_batch + batch, PERCENTILE)
	full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)

	if not full_batch:
		print("not full batch, skip and go back to iterate_batches\n")
		continue

	import torch
	obs_v = torch.FloatTensor(obs)
	acts_v = torch.LongTensor(acts)
	full_batch = full_batch[-500:]

	#print("main: from filter_batch ", "obs_v: ", obs_v, "\n")
	#time.sleep(2)

	#optimizer.zero_grad()
	optimizer = optim.Adam(params=net.parameters(), lr=0.001)

	# run the network
	print("main: run the network!\n")
	action_scores_v = net(obs_v)

	#print("From network run, action_scores_v len: ", len(action_scores_v), "data: ", action_scores_v)
	#print("acts_v len: ", len(acts_v), "data: ", acts_v)

	# calculate cross entropy loss
	loss_v = objective(action_scores_v, acts_v)
	loss_v.backward()

	optimizer.step()

	# explanations

	print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_mean, reward_bound))
	time.sleep(5) # is reward_mean increasing? (that shows convergence)

	writer.add_scalar("loss", loss_v.item(), iter_no)
	writer.add_scalar("reward_bound", reward_bound, iter_no)
	writer.add_scalar("reward_mean",  reward_mean, iter_no)

	# explanations

	if reward_mean > 199:
		print("RL Solved !!\n")
		#break

	writer.close()


'''
# original code
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
'''




