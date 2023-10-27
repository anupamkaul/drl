import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")

total_reward=0
observation, info = env.reset(seed=42)
for _ in range(1000):

    action = env.action_space.sample()
    print("action: ", action, "\n")

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print("observation: ", observation, "\n reward: ", reward, "total reward: ", total_reward, "\n")
	

    if terminated or truncated:
        print("end of episode\n")
        input()
        total_reward=0
        observation, info = env.reset()

env.close()
