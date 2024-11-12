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
import imageio

device = torch.device("cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """
    Deep Q-Network with three fully connected layers.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer providing Q-values for each action.
    """

    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        # Define the input, hidden, and output layers.
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_shape)

    def forward(self, x):
        """
        Forward pass through the network to compute Q-values for a given state.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Q-values for each possible action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
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
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.replay_buffer = deque(maxlen=int(1e5))
        input_shape = self.env.observation_space.shape[0]
        output_shape = self.env.action_space.n
        self.losses = []

        # Initialize the main and target networks, and set up the optimizer.
        self.model = DQN(input_shape, output_shape).to(device)
        self.target_model = DQN(input_shape, output_shape).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.C = 4
        self.batch_size = 64
        self.t = 0
        self.update_target_model()

    def update_target_model(self):
        """
        Copies the weights from the main model to the target model for stability in training.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def add_to_buffer(self, state, action, reward, new_state, done):
        """
        Adds a new experience to the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            new_state (np.ndarray): The next state after taking the action.
            done (bool): Whether the episode ended after this step.
        """
        state = np.array(state, dtype=np.float32)
        new_state = np.array(new_state, dtype=np.float32)
        self.replay_buffer.append((state, action, reward, new_state, done))

    def generate_action(self, state, eps):
        """
        Generates an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.
            eps (float): Epsilon value for exploration-exploitation trade-off.

        Returns:
            int: Selected action.
        """
        if np.random.random() < eps:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_target(self):
        """
        Trains the DQN using a batch of experiences from the replay buffer.
        Updates the main network and periodically syncs the target network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences.
        samples = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert data to tensors for training.
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Compute current Q-values and target Q-values.
        current_q_values = self.model(states).gather(1, actions)
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss, perform backpropagation, and update model weights.
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target model.
        self.t = (self.t + 1) % self.C
        if self.t == 0:
            self.update_target_model()

    def save_model(self, path):
        """
        Saves the current model weights to the specified file path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.model.state_dict(), "ddqn_model.pth")


def train(gamma=0.99, lr=5e-4, tau=1e-3, epsilon_decay=0.995):
    """
    Train the DQN agent in the LunarLander-v2 environment.

    Args:
        gamma (float): Discount factor for future rewards.
        lr (float): Learning rate for the optimizer.
        tau (float): Soft update parameter for the target network.
        epsilon_decay (float): Factor by which epsilon is multiplied each episode.

    Returns:
        list: Training scores for each episode.
        list: Epsilon values for each episode.
    """
    env = gym.make("LunarLander-v2")
    epsilon = 1.0  # Initial exploration rate.
    epsilon_min = 0.01  # Minimum exploration rate.
    episodes = 2000  # Total number of training episodes.
    steps = 1000  # Maximum steps per episode.

    scores = []
    scores_window = deque(maxlen=100)  # Store recent scores to track progress.
    epsilons = []
    agent = DQNAgent(env=env, gamma=gamma, lr=lr, tau=tau)

    for trial in range(episodes):
        score = 0
        cur_state = env.reset()
        epsilons.append(epsilon)

        # Interact with the environment for a set number of steps.
        for step in range(steps):
            action = agent.generate_action(cur_state, epsilon)
            new_state, reward, done, info = env.step(action)
            done = done
            score += reward
            agent.add_to_buffer(cur_state, action, reward, new_state, done)
            agent.train_target()
            cur_state = new_state

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        print('\rTrial {}\tMean Score: {:.3f}\tEpsilon: {:.3f}'.format(trial, np.mean(scores_window), epsilon), end="")

        if trial % 100 == 0:
            print('\rTrial {}\tMean Score: {:.3f}'.format(trial, np.mean(scores_window)))

        # Save the model if a mean score threshold is reached.
        if np.mean(scores_window) >= 200.0:
            agent.save_model("dqn_model.pth")
            print('\nAchieved Mean Score of 200 for past 100 trials at trial {:d}!\tAverage Score: {:.3f}'.format(
                trial - 100, np.mean(scores_window)))
            break

    env.close()
    plt.figure()
    plt.plot(agent.losses)
    plt.title("DDQN Training Loss Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("DDQN_Training_Loss.png")
    plt.show()
    return scores, epsilons


def test(saved_model):
    """
    Test the trained DQN model in the LunarLander-v2 environment and record gameplay as a GIF.

    Args:
        saved_model (str): Path to the saved model file.

    Returns:
        list: Test scores for each episode.
    """
    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env=env)
    agent.model.load_state_dict(torch.load(saved_model))
    agent.model.eval()

    scores = []
    frames = []  # For recording frames for the GIF.
    for trial in range(100):
        score = 0
        cur_state = env.reset()
        for step in range(1000):
            state = torch.FloatTensor(cur_state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(agent.model(state)).item()
            new_state, reward, done, info = env.step(action)
            done = done
            score += reward
            cur_state = new_state

            # Record frames for the GIF only if the score is high (e.g., after 95 trials).
            if trial >= 95:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if done:
                break

        scores.append(score)
        print('\rTrial {}\tScore: {:.3f}'.format(trial, score), end="")

    env.close()

    gif_path = "dqn_lunar_lander.gif"
    #imageio.mimsave(gif_path, frames, fps=30)
    print(f"\nGIF saved at {gif_path}")
    return scores


def plot_values(values, xlabel, ylabel, title=None, save_as=None):
    """
    Plot values over episodes (e.g., scores, epsilon).

    Args:
        values (list): List of values to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str, optional): Title of the plot.
    """
    plt.figure()
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if save_as:
        plt.savefig(save_as)
    plt.show()


train_scores, epsilons = train()

plot_values(train_scores, "Episodes", "Scores", title="DDQN Training Scores", save_as="DDQN_Training_Scores.png")

mean_scores = [np.mean(train_scores[max(0, i - 99):i + 1]) for i in range(len(train_scores))]
plot_values(mean_scores, "Episodes", "Mean Scores", title="DDQN Mean Scores Over Last 100 Episodes", save_as="DDQN_Mean_Scores.png")

plot_values(epsilons, "Episodes", "Epsilon", title="DDQN Epsilon Decay Over Episodes", save_as="DDQN_Epsilon_Decay.png")

test_scores = test("ddqn_model.pth")

plot_values(test_scores, "Episodes", "Scores", title="DDQN Test Scores", save_as="DDQN_Test_Scores.png")

pickle.dump(train_scores, open("ddqn_train_scores.p", "wb"))
pickle.dump(test_scores, open("ddqn_test_scores.p", "wb"))
pickle.dump(epsilons, open("ddqn_epsilons.p", "wb"))