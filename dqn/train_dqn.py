import numpy as np
np.bool8 = bool  # for gym compatibility

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# Initialize environment
env = gym.make("FrozenLake-v1", is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

# One-hot encode discrete state
def one_hot(state, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[state] = 1.0
    return vec

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.net(x)

# Initialize policy and target networks
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Replay memory
memory = deque(maxlen=2000)
batch_size = 64

# Hyperparameters
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
episodes = 2000
rewards = []

# Select action with ε-greedy policy
def select_action(state_vec):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        qs = policy_net(state_tensor)
        return int(torch.argmax(qs).item())

# Experience replay
def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    
    states, actions, rewards_b, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards_b = torch.FloatTensor(rewards_b)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    current_qs = policy_net(states).gather(1, actions).squeeze()
    next_qs = target_net(next_states).max(1)[0].detach()
    targets = rewards_b + gamma * next_qs * (~dones)

    loss = criterion(current_qs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for ep in range(episodes):
    state_raw, _ = env.reset()
    state = one_hot(state_raw, state_size)
    total_reward = 0
    done = False

    while not done:
        action = select_action(state)
        next_raw, reward, done, truncated, _ = env.step(action)
        next_state = one_hot(next_raw, state_size)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if ep % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards.append(total_reward)

    # Show progress
    if ep % 100 == 0:
        avg = np.mean(rewards[-100:])
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Avg(100): {avg:.3f}")

# Save model and results
os.makedirs("results", exist_ok=True)
torch.save(policy_net.state_dict(), "results/dqn_model.pth")
np.save("results/dqn_rewards.npy", rewards)

# Plot rewards
smoothed = np.convolve(rewards, np.ones(100)/100, mode="valid")
plt.plot(smoothed)
plt.title("DQN Performance")
plt.xlabel("Episode")
plt.ylabel("Avg Reward (100-episode window)")
plt.grid()
plt.savefig("results/dqn_plot.png")
plt.show()

print("Training complete! Model and plot saved in 'results/' folder.")
