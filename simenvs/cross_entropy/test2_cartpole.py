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
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
	def forward(self, x):
		return self.net(x)

'''
Above is a pretty simple network: it takes a single observation from the environment as an input vector
and outputs a number for every action that we can perform. The output from the network is a probability
distribution over actions. Instead of including the traditional softmax non-linearity after the last step,
(which uses exponentiation) and then calculating cross-entropy loss (which uses logarithm of probabilities)
we'll use the pytorch class nn.CrossEntropyLoss which combines softmax and cross-entropy in a single, more
numerically stable expression. 

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
	obs = env.reset()
	sm = nn.Softmax(dim=1)

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

	while True:
		obs_v = torch.FloatTensor([obs])
		act_probs_v = sm(net(obs_v))     #apply softmax to the neural network (agent) that takes in 1 observation
		act_probs = act_probs_v.data.numpy()[0]
	
		# explanation of above

		action = np.random.choice(len(act_probs), p = act_probs)
		next_obs, reward, is_done, _ = env.step(action)

		# write some more

		episode_reward += reward
		episode_steps.append(EpisodeStep(observation=obs, action=action))

		if is_done:
			batch.append(Episode(reward=episode_reward, steps = episode_steps))	
			episode_reward = 0.0
			episode_steps = []
			next_obs = env.reset()

			#exit loop
			if len(batch) == batch_size:
				yield batch

				batch = []
				obs = next_obs


def filter_batch(batch, percentile):

	rewards = list(map(lambda s: s.reward, batch))
	reward_bound = np.percentile(rewards, percentile)
	reward_mean = float(np.mean(rewards))

	#explanations galore!

	train_obs = []
	train_act = []

	for example in batch:
		if example.reward < reward_bound:
			continue
		train_obs.extend(map(lambda step: step.action, example.steps))

		#explanations

	train_obs_v = torch.FloatTensor(train_obs)
	train_act_v = torch.LongTensor(train_act)
	return train_obs_v, train_act_v, reward_bound, reward_mean

# main glue (continues from top)

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

net = Net(obs_size, HIDDEN_SIZE, n_actions)
objective = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(params=net.parameters(), lr = 0.01)

from tensorboardX import SummaryWriter
writer = SummaryWriter()

#explanations

for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
	obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
	optimizer.zero_grad()
	action_scores_v = net(obs_v)
	loss_v = objective(action_scores_v, acts_v)
	loss_v.backward()
	optimizer.step()

	# explanations

	print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward<b))
	writer.add_scalar("loss", loss_v.item(), iter_no)
	writer.add_scalar("reward_bound", reward_b, iter_no)
	writer.add_scalar("reward_mean",  reward_m, iter_no)

	# explanations

	if reward_m > 199:
		print("RL Solved !!\n")
		break

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



