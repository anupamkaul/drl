#!/usr/bin/env python3
#import gym
import gymnasium as gym
import collections
from tensorboardX import SummaryWriter

#ENV_NAME = "FrozenLake-v0"
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

import time

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset(seed=42)

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

            #unhashable because env now contains state transition probability expressed as a dict itself inside state

            #self.rewards[(self.state, action, new_state)] = reward

            # original code break for introspection...
            # quick hack to find out the offending dict type within the tuple self.state
            print("\nintrospecting self.state..")
            #print("self.state: ", self.state, " type of self.state" , type(self.state), " number of elements: ", len(self.state), "\n") 
            print("new_state: ", new_state, " type of new.state", type(new_state))
            i = 0
            print("count: ", _, "\n")

            '''
            print("\nintrospecting self.state tuple..")
            while i < len(self.state):
                print("type: ", type(self.state[i]), "val: ", self.state[i])
                i = i + 1

            print("\nintrospectng the dict element in the tuple..")
            for k, v in self.state[1].items():
                print("key ", k, " value ", v, "\n") # this reveals the 'name' of the key in the dict..

            rewards_key = (self.state[0], self.state[1]["prob"],  action,  new_state)
            #rewards_key = (self.state[0], self.state[1]["prob"],  action,  new_state[0], newstate[1]["prob"])
            print("rewards key: ", rewards_key, "\n")

            #self.rewards[(self.state[0], self.state[1]["prob"], action, new_state)] = reward
            print("\npass the self.rewards hash assigment of reward\n")

            # original code continue...
            #self.transits[(self.state, action)][new_state] += 1
            # self.transits[(self.state[0], self.state[1]["prob"], action)][new_state] += 1
            print("\npass the self.transits hash assigment of transits\n")
            '''
            
            self.state = self.env.reset(seed=42) if is_done else new_state
           
            print("after env.reset self.state: ", self.state, " type of self.state" , type(self.state), "\n") 
            time.sleep(2)
            
            #print("after env.reset self.state: ", self.state, " type of self.state" , type(self.state), " number of elements: ", len(self.state), "\n") 

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
        total_reward = 0.0
        state = env.reset(seed=42)
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
