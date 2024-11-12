# LunarLander-v2 PPO Implementation

This repository contains a PPO (Proximal Policy Optimization) implementation for the `LunarLander-v2` environment, designed to train an agent to effectively land in the specified zone.

## Files

- `LunarLander-v2_PPO.zip`: Contains the main code and associated files for the PPO implementation.
  - `fast.py`: Trains the PPO model using a multiprocessed approach for faster training. This is the quickest way to train the agent in the environment.
  - `LunarLander-v2_PPO.py`: Trains the PPO model episode by episode, allowing for evaluation of performance after each episode.

## Requirements

To run this code, ensure you have the necessary dependencies. You can install them by running:

```bash
pip install -r requirements.txt
