# Q-Learning for Frozen Lake
import numpy as np
np.bool8 = bool

import gym
import matplotlib.pyplot as plt
# … rest of your existing code …

import numpy as np
import gym
import matplotlib.pyplot as plt
import os

env = gym.make("FrozenLake-v1", is_slippery=True)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 2000
rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, _ = env.step(action)
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

# Save Q-table and plot
os.makedirs("results", exist_ok=True)
np.save("results/q_table.npy", q_table)
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title("Q-Learning Performance")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid()
plt.savefig("results/q_learning_plot.png")
plt.show()
