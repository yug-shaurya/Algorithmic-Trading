# Algorithmic Trading with Reinforcement Learning

Developed as part of the  **IIT Bombay Summer of Science (SoS) Program**.

## Overview

This project explores the application of Reinforcement Learning (RL) to build autonomous trading agents that learn buy/hold/sell strategies from historical market data — without explicitly programmed rules.

Two RL approaches are implemented and compared:

- **Q-Learning** — A tabular RL method where the agent learns a discrete action-value function over a discretized state space
- **Deep Q-Network (DQN)** — A neural network-based agent that approximates the Q-function, allowing for richer, continuous state representations

## Problem Formulation

| Component | Description |
|-----------|-------------|
| **State** | Recent price returns, technical indicators (e.g. RSI, moving averages), current position |
| **Actions** | Buy / Hold / Sell |
| **Reward** | Change in portfolio value (PnL) per timestep |
| **Environment** | Simulated trading on historical OHLCV data |

## Project Structure
```
Algorithmic-Trading/
├── q_learning/       # Tabular Q-Learning agent
├── dqn/              # Deep Q-Network agent
└── README.md
```

## Key Learnings & Challenges

- **Non-stationarity**: Financial markets are non-stationary — strategies that work in one regime often fail in another
- **Reward design**: Naive PnL rewards incentivize high-risk behavior; risk-adjusted rewards (e.g. Sharpe ratio) are more robust
- **Overfitting**: RL agents can memorize historical patterns without generalizing — rigorous train/test splits are essential

## Future Work

- Incorporate Sharpe-ratio-based reward shaping for risk-adjusted learning
- Test on Indian equity data (Nifty 50)
- Add proper backtesting metrics: drawdown, win rate, Calmar ratio

## Tech Stack

- Python, NumPy, Pandas
- PyTorch / TensorFlow (for DQN)
- Matplotlib (for visualizations)

## Author

**Yugratna Shaurya**  - Student, IIT Bombay  
Summer of Science 2024-25, IIT Bombay
