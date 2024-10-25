#!/usr/bin/env python
# coding: utf-8

"""
Author: Liu Xingyan
Date: October 25, 2024
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import pandas as pd


class DQNNetwork(nn.Module):
    """
    Deep Q-Network (DQN) using a simple feedforward neural network.

    Attributes:
        input_shape (int): The dimensionality of the input (state size).
        output_shape (int): The dimensionality of the output (number of actions).
    """

    def __init__(self, input_shape, output_shape):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_shape)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """
    DQN agent that manages experience replay, training, and interaction with the environment.

    Attributes:
        env (gym.Env): The environment in which the agent interacts.
        gamma (float): Discount factor for future rewards.
        lr (float): Learning rate for the optimizer.
        tau (float): Soft update parameter for target network updates.
        replay_buffer (deque): Experience replay buffer.
        model (DQN): Main DQN model for learning.
        target_model (DQN): Target DQN model for stability in training.
        optimizer (optim.Adam): Optimizer for the model.
        C (int): Frequency of target network updates.
        batch_size (int): Batch size for training.
        t (int): Counter for updating the target network.
    """

    def __init__(self, env, gamma=0.99, lr=5e-4, tau=1e-3):
        self.env = env
        self.replay_buffer = deque(maxlen=int(1e5))
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = 64
        self.C = 4
        self.t = 0

        input_shape = self.env.observation_space.shape[0]
        output_shape = self.env.action_space.n

        # Create model and target model
        self.model = DQNNetwork(input_shape, output_shape)
        self.target_model = DQNNetwork(input_shape, output_shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Copy parameters from model to target_model
        self.update_target_model()

    def update_target_model(self):
        """
        Update the target model's parameters by copying from the main model.
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update_target(self):
        """
        Perform a soft update of the target network using the main model's parameters.
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_buffer(self, state, action, reward, new_state, done):
        """
        Add a transition to the replay buffer.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            new_state (np.array): The next state.
            done (bool): Whether the episode ended.
        """
        self.replay_buffer.append((state, action, reward, new_state, done))

    def generate_action(self, state, eps):
        """
        Generate an action using an epsilon-greedy policy.

        Args:
            state (np.array): The current state.
            eps (float): The probability of selecting a random action.

        Returns:
            int: The chosen action.
        """
        if np.random.random() < eps:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def learning(self):
        """
        Sample a batch from the replay buffer and perform a learning step.

        This involves calculating the loss between predicted Q-values and target Q-values,
        followed by a backward pass to update model parameters.
        """
        self.t = (self.t + 1) % self.C
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.t != 0:
            return

        samples = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(np.array([np.array(state, dtype=np.float32) for state in states]))
        next_states = torch.FloatTensor(np.array([np.array(state, dtype=np.float32) for state in next_states]))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target()

    def save_model(self, fn):
        """
        Save the trained model's state to a file.

        Args:
            fn (str): The file name for saving the model.
        """
        torch.save(self.model.state_dict(), fn)


# Training function
def train(gamma=0.99, lr=5e-4, tau=1e-3, epsilon_decay=0.995):
    """
    Train the DQN agent on the LunarLander-v2 environment.

    Args:
        gamma (float): Discount factor.
        lr (float): Learning rate.
        tau (float): Soft update parameter.
        epsilon_decay (float): Rate of decay for epsilon (exploration rate).

    Returns:
        list: Scores obtained during training.
        list: Epsilon values used during training.
    """
    env = gym.make("LunarLander-v2")
    epsilon = 1.0
    epsilon_min = 0.01
    episodes = 1000
    steps = 500
    scores = []
    scores_window = deque(maxlen=100)
    epsilons = []

    agent = DQN(env=env, gamma=gamma, lr=lr, tau=tau)

    for trial in range(episodes):
        score = 0
        cur_state = env.reset()
        if isinstance(cur_state, tuple):
            cur_state = cur_state[0]
        cur_state = np.array(cur_state, dtype=np.float32)
        epsilons.append(epsilon)

        for step in range(steps):
            action = agent.generate_action(cur_state, epsilon)
            new_state, reward, done, truncated, _ = env.step(action)
            if isinstance(new_state, tuple):
                new_state = new_state[0]
            new_state = np.array(new_state, dtype=np.float32)
            score += reward
            agent.add_to_buffer(cur_state, action, reward, new_state, done)
            agent.learning()
            cur_state = new_state
            if done or truncated:
                break

        scores.append(score)
        scores_window.append(score)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f'\rtrial {trial} Mean Score: {np.mean(scores_window):.3f} with epsilon: {epsilon:.3f}', end="")
        if trial % 100 == 0:
            print(f'\rtrial {trial} Mean Score: {np.mean(scores_window):.3f}')

        if np.mean(scores_window) >= 200.0:
            print(f'\nAchieved Mean Score of 200 over past 100 trials in {trial} episodes.')
            break

    env.close()
    return scores, epsilons


def plot_values(values, xlabel, ylabel):
    """
    Plot values over episodes.

    Args:
        values (list): List of values to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure()
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Hyperparameter tuning
scores_lr = {}
lrs = [5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

for lr in lrs:
    print(f"\nlearning rate value: {lr}\n", end="")
    scores_lr[lr], _ = train(lr=lr)

pickle.dump(scores_lr, open("scores_lr.p", "wb"))

scores_lr_pd = pd.DataFrame(list(scores_lr[5e-5]), columns=[f"lr={5e-5}"])

for lr in lrs[1:]:
    df = pd.DataFrame(list(scores_lr[lr]), columns=[f"lr={lr}"])
    scores_lr_pd = pd.concat([scores_lr_pd, df], axis=1)

scores_lr_pd['Trials'] = np.arange(scores_lr_pd.shape[0])
print(scores_lr_pd)

plt.plot("Trials", "lr=5e-05", data=scores_lr_pd)
plt.plot("Trials", "lr=0.0005", data=scores_lr_pd)
plt.plot("Trials", "lr=0.005", data=scores_lr_pd)
plt.plot("Trials", "lr=0.05", data=scores_lr_pd)
plt.plot("Trials", "lr=0.5", data=scores_lr_pd)
plt.legend()
plt.ylabel("Scores")
plt.xlabel("Trials")
plt.show()