import torch
import gym
import numpy as np
from train_dqn import DQN, one_hot

from train_dqn import one_hot

# Set up environment
env = gym.make("FrozenLake-v1", is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

# Load trained model
policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load("results/dqn_model.pth"))
policy_net.eval()

# Evaluation loop
episodes = 100
wins = 0

def select_action_eval(state):
    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32)
        q_values = policy_net(state_tensor)
        return q_values.argmax().item()

for _ in range(episodes):
    state_raw, _ = env.reset()
    state = one_hot(state_raw, state_size)
    done = False
    while not done:
        action = select_action_eval(state)
        next_state_raw, reward, done, truncated, _ = env.step(action)
        state = one_hot(next_state_raw, state_size)
        if done and reward == 1.0:
            wins += 1

print(f"\n✅ DQN Win Rate over {episodes} episodes: {wins}%\n")

