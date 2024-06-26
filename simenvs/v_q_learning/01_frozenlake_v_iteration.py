#!/usr/bin/env python3
#import gym
import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

#import time

class Agent:
    def __init__(self):

        # check: does make go via register, because register sets max_steps at 4*4 to 100 but it doesn't work
        self.env = gym.make(ENV_NAME, is_slippery=False, render_mode="human", max_episode_steps=100)

        self.state = self.env.reset(seed=42)
        #print("INIT : self.state: ", self.state, " type of self.state" , type(self.state), "\n") 
        self.state = 0 # override, because it anyways returns self.state:  (0, {'prob': 1})  type of self.state <class 'tuple'>

	#rewards table
        self.rewards = collections.defaultdict(float)

	# transition table
        self.transits = collections.defaultdict(collections.Counter)

	# value table
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, info = self.env.step(action)

            print("\rplay_n_random_steps: count: ", _, end= ' ',flush=True)

            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1

            if is_done:
                self.state = self.env.reset(seed=42)
                self.state = 0 # override, because it anyways returns self.state:  (0, {'prob': 1})  type of self.state <class 'tuple'>
            else:
                self.state = new_state

        import time
        print("at end of play_n_random_steps: self.rewards = ", self.rewards, "\n there are ", len(self.rewards), " unique <s,a,s> that were mapped" )
        time.sleep(2)
        print("at end of play_n_random_steps: self.transits = ", self.transits, "\n there are ", len(self.transits), " unique transits that were mapped")
        time.sleep(2)
           

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        print("agent: play episode")
        total_reward = 0.0
        state = env.reset(seed=42)
        state = 0 # override, because it anyways returns self.state:  (0, {'prob': 1})  type of self.state <class 'tuple'>
        episode_counter=0
 
        # restricting to 100 steps doesn't work, need to check if code goes via registry, disable for now
        #env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

        while True:
            action = self.select_action(state)
            #print("\nnew_action: ", action)

            # test start
            #action = 2 # just go right
            # randomize action instead of selection the best, just to see if play_episode ends
            action = self.env.action_space.sample()
            # test end           

            new_state, reward, is_done, truncated, info = env.step(action)
            #print("new_state: ", new_state)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            episode_counter=episode_counter+1
            print("\rplay_episode: total rewards: ", total_reward, "counter: ", episode_counter, end = ' ', flush=True)
            if is_done:
                print("BREAK: We got Is_Done\n")
                break
            state = new_state
        
        print("WE RETURN TOTAL REWARD", total_reward, "\n")
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    import time
    test_env = gym.make(ENV_NAME, is_slippery=False, render_mode="human", max_episode_steps=100) # why are we creating 2 envs, one here and one inside the agent class ??
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        print("begin agent's value iteration")
        agent.value_iteration()
        print("end agent's value iteration")

        reward = 0.0
        print("Enter TEST_EPISODES")
        for _ in range(TEST_EPISODES):
            print("Call play_episode ", _ , " of ", TEST_EPISODES, "\n")
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        print("%d Reward is now %.3f\n" % (iter_no, reward))
        time.sleep(2)
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
