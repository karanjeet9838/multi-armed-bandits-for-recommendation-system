# Multi-Armed Bandits for Recommendation Systems

A Python implementation of various Multi-Armed Bandit (MAB) algorithms for building intelligent recommendation systems. This project demonstrates different exploration-exploitation strategies and their performance characteristics.

## Overview

Multi-Armed Bandits are a fundamental approach in machine learning for balancing **exploration** (trying new options) and **exploitation** (choosing known good options). This repository implements several MAB algorithms and evaluates them using offline evaluation techniques.

## Features

The notebook implements and evaluates the following algorithms:

1. **ε-Greedy (Epsilon-Greedy)**
   - Simple exploration strategy with fixed exploration probability
   - Explores with probability ε, exploits with probability 1-ε

2. **UCB (Upper Confidence Bound)**
   - Optimistic approach using confidence bounds
   - Balances exploration and exploitation automatically
   - Better theoretical guarantees than ε-greedy

3. **Additional Contextual Bandits**
   - Decision Tree-based context-aware bandits
   - RBF kernel-based approaches for complex interactions

## Project Structure

```
├── MABs.ipynb          # Main notebook with algorithm implementations
├── dataset.txt         # Historical interaction dataset
└── README.md          # This file
```

## Files

- **MABs.ipynb**: Jupyter notebook containing:
  - Base MAB class with abstract methods
  - Multiple algorithm implementations
  - Offline evaluation framework
  - Performance comparisons and visualizations

- **dataset.txt**: Historical log of user interactions with format:
  - Column 1: Selected arm (recommendation)
  - Column 2: Reward (user feedback)
  - Columns 3+: Context features

## Key Components

### Base MAB Class
Abstract base class defining the interface for all MAB algorithms:
- `play(context)`: Select an arm to play
- `update(arm, reward, context)`: Update internal state based on feedback

### Offline Evaluation
`offlineEvaluate()` function to evaluate MAB algorithms on historical data:
- Simulates online learning from historical logs
- Counts matching events (where algorithm would have chosen logged arm)
- Computes cumulative rewards

## Usage

1. Open `MABs.ipynb` in Jupyter Notebook or JupyterLab
2. Run the cells sequentially to:
   - Load the dataset
   - Initialize different MAB algorithms
   - Evaluate their performance offline
   - Compare results

### Example
```python
# Initialize an ε-greedy bandit with 10 arms and ε=0.05
mab = EpsGreedy(n_arms=10, epsilon=0.05)

# Evaluate on historical data
results = offlineEvaluate(mab, arms, rewards, contexts, n_rounds=800)
print(f'Average reward: {np.mean(results)}')
```

## Key Concepts

### Exploration vs Exploitation
- **Exploration**: Trying new arms to learn about their potential
- **Exploitation**: Choosing the best-known arm to maximize immediate reward
- MAB algorithms balance these two competing objectives

### Contextual Bandits
- Extend basic bandits by using context (features) to make better decisions
- Different arms may be optimal for different contexts

### Offline Evaluation
- Evaluates algorithms on historical data without deploying them
- Uses importance sampling concepts to avoid bias

## Requirements

- numpy
- scikit-learn
- matplotlib

## Algorithms Comparison

Each algorithm has different properties:

| Algorithm | Exploration | Theoretical Guarantee | Adaptive |
|-----------|------------|----------------------|----------|
| ε-Greedy | Fixed probability | No | No |
| UCB | Automatic | Yes (Regret bounds) | Yes |
| Contextual | Context-dependent | Varies | Yes |

## Performance Metrics

- **Average Reward**: Mean reward per play
- **Cumulative Reward**: Total reward over evaluation period
- **Convergence**: How quickly the algorithm learns optimal arms

## References

Key concepts in this implementation:
- Multi-Armed Bandits provide solutions to the exploration-exploitation tradeoff
- Contextual information improves recommendation quality
- Offline evaluation enables safe algorithm testing

#
