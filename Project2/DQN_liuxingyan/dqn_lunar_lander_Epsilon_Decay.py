#!/usr/bin/env python
# coding: utf-8

"""
Author: Liu Xingyan
October 25, 2024
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
import pickle

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes the Q-network with two hidden layers.
        Args:
            input_dim (int): Dimension of the input layer (state space).
            output_dim (int): Dimension of the output layer (action space).
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass through the Q-network.
        Args:
            x (torch.Tensor): Input state.
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
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.replay_buffer = deque(maxlen=int(1e5))
        self.batch_size = 64
        self.C = 4
        self.t = 0

        # Get dimensions of input (state) and output (actions)
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        # Initialize Q-network and target network
        self.model = QNetwork(input_dim, output_dim)
        self.target_model = QNetwork(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Set up the optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def add_to_buffer(self, state, action, reward, new_state, done):
        """
        Adds a new experience to the replay buffer.
        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            new_state (np.array): Next state.
            done (bool): Whether the episode ended.
        """
        state = np.array(state, dtype=np.float32).flatten()
        new_state = np.array(new_state, dtype=np.float32).flatten()
        self.replay_buffer.append((state, action, reward, new_state, done))

    def generate_action(self, state, eps):
        """
        Generates an action using an epsilon-greedy policy.
        Args:
            state (np.array): Current state.
            eps (float): Epsilon value for exploration.
        Returns:
            int: Selected action.
        """
        if random.random() < eps:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_target(self):
        """
        Softly updates the target network parameters using tau.
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learning(self):
        """
        Samples a batch from the replay buffer and updates the Q-network.
        """
        self.t = (self.t + 1) % self.C
        if self.t != 0 or len(self.replay_buffer) < self.batch_size:
            return # Skip training if not enough experiences

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert sampled experiences into tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute the current Q-values and the target Q-values
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and backpropagate to update the model
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_target()

    def save_model(self, fn):
        """
        Saves the model's parameters to a file.
        Args:
            fn (str): Filename to save the model parameters.
        """
        torch.save(self.model.state_dict(), fn)

# Function for training the agent
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
        cur_state, _ = env.reset()  # Updated to handle new reset API
        cur_state = np.array(cur_state, dtype=np.float32).flatten()  # Ensure state is flattened
        epsilons.append(epsilon)

        for step in range(steps):
            action = agent.generate_action(cur_state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = np.array(new_state, dtype=np.float32).flatten()  # Ensure new_state is flattened
            done = terminated or truncated
            score += reward
            agent.add_to_buffer(cur_state, action, reward, new_state, done)
            agent.learning()
            cur_state = new_state

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f'\rTrial {trial} Mean Score: {np.mean(scores_window):.2f} Epsilon: {epsilon:.2f}', end="")
        if trial % 100 == 0:
            print(f'\rTrial {trial} Mean Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 200.0:
            print(f'\nSolved in {trial} trials with Average Score: {np.mean(scores_window):.2f}')
            break

    env.close()
    return scores, epsilons

def plot_values(values, xlabel, ylabel):
    plt.figure()
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

scores_epsilon_decay = {}
epsilons_epsilon_decay = {}
epsilon_decays = [0.8, 0.9, 0.99, 0.995, 1]

for epsilon_decay in epsilon_decays:
    print(f"\nEpsilon Decay Ratio: {epsilon_decay}")
    scores_epsilon_decay[epsilon_decay], epsilons_epsilon_decay[epsilon_decay] = train(epsilon_decay=epsilon_decay)

pickle.dump(scores_epsilon_decay, open("scores_epsilon_decay.p", "wb"))
pickle.dump(epsilons_epsilon_decay, open("epsilons_epsilon_decay.p", "wb"))

scores_epsilon_decay_pd = pd.DataFrame(scores_epsilon_decay)
scores_epsilon_decay_pd['Trials'] = np.arange(scores_epsilon_decay_pd.shape[0])

plt.figure()
for decay in epsilon_decays:
    plt.plot(scores_epsilon_decay_pd['Trials'], scores_epsilon_decay_pd[decay], label=f"epsilon_decay={decay}")
plt.xlabel("Trials")
plt.ylabel("Scores")
plt.legend()
plt.show()

epsilons_epsilon_decay_pd = pd.DataFrame(epsilons_epsilon_decay)
epsilons_epsilon_decay_pd['Trials'] = np.arange(epsilons_epsilon_decay_pd.shape[0])

plt.figure()
for decay in epsilon_decays:
    plt.plot(epsilons_epsilon_decay_pd['Trials'], epsilons_epsilon_decay_pd[decay], label=f"epsilon_decay={decay}")
plt.xlabel("Trials")
plt.ylabel("Epsilon")
plt.legend()
plt.show()