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
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import pandas as pd


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        Initializes a feed-forward neural network with two hidden layers.

        Args:
            input_shape (int): Dimension of the input layer (state space).
            output_shape (int): Dimension of the output layer (action space).
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_shape)

    def forward(self, x):
        """
        Forward pass through the DQN network.

        Args:
            x (torch.Tensor): Input tensor (state).

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, env, gamma=0.99, lr=5e-4, tau=1e-3):
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
        self.env = env
        self.replay_buffer = deque(maxlen=int(1e5))
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = 64
        self.C = 4
        self.t = 0

        # Define the input and output dimensions based on the environment
        input_shape = self.env.observation_space.shape[0]
        output_shape = self.env.action_space.n

        # Initialize the Q-network and the target Q-network
        self.model = DQNNetwork(input_shape, output_shape)
        self.target_model = DQNNetwork(input_shape, output_shape)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Initialize the target network with the same parameters as the model
        self.update_target_model()

    def update_target_model(self):
        """
        Updates the target network by copying weights from the Q-network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update_target(self):
        """
        Softly updates the target network parameters using the formula:
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_buffer(self, state, action, reward, new_state, done):
        """
        Adds a transition (state, action, reward, next state, done) to the replay buffer.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            new_state (np.array): Next state.
            done (bool): Whether the episode ended.
        """
        self.replay_buffer.append((state, action, reward, new_state, done))

    def generate_action(self, state, eps):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.array): Current state.
            eps (float): Epsilon value for exploration.

        Returns:
            int: Selected action.
        """
        if np.random.random() < eps:
            return self.env.action_space.sample()  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()  # Exploit

    def learning(self):
        """
        Performs learning on the Q-network by sampling from the replay buffer.
        Updates the model parameters and performs a soft update on the target network.
        """
        self.t = (self.t + 1) % self.C
        if len(self.replay_buffer) < self.batch_size or self.t != 0:
            return  # Skip training if not enough samples or not time for target update

        # Sample a batch of experiences from the replay buffer
        samples = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert data into tensors
        states = torch.FloatTensor(np.array(states)).squeeze(1)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = torch.FloatTensor(np.array(next_states)).squeeze(1)
        dones = torch.FloatTensor(dones).view(-1, 1)

        # Compute current Q-values and target Q-values
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].view(-1, 1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss using Mean Squared Error
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform a soft update on the target network
        self.soft_update_target()

    def save_model(self, filename):
        """
        Saves the trained Q-network's parameters.

        Args:
            filename (str): File path to save the model.
        """
        torch.save(self.model.state_dict(), filename)


def train(gamma=0.99, lr=5e-4, tau=1e-3, epsilon_decay=0.995):
    """
    Trains the DQN agent in the LunarLander-v2 environment.

    Args:
        gamma (float): Discount factor for rewards.
        lr (float): Learning rate for the optimizer.
        tau (float): Soft update parameter for the target network.
        epsilon_decay (float): Decay rate for epsilon (exploration).

    Returns:
        tuple: List of scores and list of epsilon values for each episode.
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
        # Extract initial state from the environment (Gym's new API)
        cur_state, _ = env.reset()
        cur_state = cur_state.reshape(1, -1)
        epsilons.append(epsilon)

        for step in range(steps):
            action = agent.generate_action(cur_state, epsilon)
            new_state, reward, done, truncated, _ = env.step(action)
            score += reward
            new_state = new_state.reshape(1, -1)
            agent.add_to_buffer(cur_state, action, reward, new_state, done or truncated)
            agent.learning()
            cur_state = new_state
            if done or truncated:
                break

        scores.append(score)
        scores_window.append(score)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

        # Output progress and performance
        print(f'\rTrial {trial} | Mean Score: {np.mean(scores_window):.3f} | Epsilon: {epsilon:.3f}', end="")
        if trial % 100 == 0:
            print(f'\rTrial {trial} | Mean Score: {np.mean(scores_window):.3f}')

        # Check if the agent has solved the environment
        if np.mean(scores_window) >= 200.0:
            print(f'\nAchieved Mean Score of 200 over the past 100 trials in {trial} episodes.')
            break

    env.close()
    return scores, epsilons


def plot_values(values, xlabel, ylabel):
    plt.figure()
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


scores_gamma = {}
gammas = np.linspace(0.5, 0.99, num=5)

for gamma in gammas:
    print(f"\nTraining with gamma value: {gamma:.3f}\n")
    scores_gamma[gamma], _ = train(gamma=gamma)

with open("scores_gamma.p", "wb") as f:
    pickle.dump(scores_gamma, f)

scores_gamma_pd = pd.DataFrame(list(scores_gamma[0.5]), columns=[f"Gamma={0.5}"])

for gamma in gammas[1:]:
    df = pd.DataFrame(list(scores_gamma[gamma]), columns=[f"Gamma={gamma}"])
    scores_gamma_pd = pd.concat([scores_gamma_pd, df], axis=1)

scores_gamma_pd['Trials'] = np.arange(scores_gamma_pd.shape[0])

for gamma in gammas:
    plt.plot("Trials", f"Gamma={gamma}", data=scores_gamma_pd, label=f"Gamma={gamma}")
plt.legend()
plt.ylabel("Scores")
plt.xlabel("Trials")
plt.show()