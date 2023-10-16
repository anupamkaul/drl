import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    print("action: ", action, "\n")

    observation, reward, terminated, truncated, info = env.step(action)
    print("observation: ", observation, "\n reward: ", reward, "\n")
	

    if terminated or truncated:
        observation, info = env.reset()
env.close()
