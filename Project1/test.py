# -*- coding: utf-8 -*-
"""
Author: Xingyan Liu
Date: September 26, 2024
"""
import numpy as np
from matplotlib import pyplot as plt


# Get neighboring states
def state_nei(state):
    neighbors = {'B': ['A', 'C'],
                 'C': ['B', 'D'],
                 'D': ['C', 'E'],
                 'E': ['D', 'F'],
                 'F': ['E', 'G']}
    return neighbors[state]


# Convert state to vector representation
def state_to_vec(state):
    vectors = {'B': np.array([1, 0, 0, 0, 0]),
               'C': np.array([0, 1, 0, 0, 0]),
               'D': np.array([0, 0, 1, 0, 0]),
               'E': np.array([0, 0, 0, 1, 0]),
               'F': np.array([0, 0, 0, 0, 1])}
    return vectors[state]


# Generate sequence of states and their vector representations
def generate_sequence():
    state = 'D'
    seq = [state]
    seq_vec = [state_to_vec(state)]
    reward = 0
    while state not in ['A', 'G']:
        state = np.random.choice(state_nei(state))
        seq.append(state)
        if state not in ['A', 'G']:
            seq_vec.append(state_to_vec(state))
        reward = 1 if state == 'G' else 0
    return seq, seq_vec, reward


# Generate multiple training sets
def generate_trainingsets(num_training_sets, num_seqs):
    training_sets, reward_sets = [], []
    for _ in range(num_training_sets):
        training_seqs, reward_seqs = [], []
        for _ in range(num_seqs):
            _, seq_vec, reward = generate_sequence()
            training_seqs.append(seq_vec)
            reward_seqs.append(reward)
        training_sets.append(training_seqs)
        reward_sets.append(reward_seqs)
    return training_sets, reward_sets


# Perform TD learning on one set of training sequences
def td_training_seqs(training_seqs, reward_seqs, lambd, alpha=0.01, epsilon=0.001):
    w = np.ones(5) * 0.5
    while True:
        w_old = w.copy()
        for seqs, rew in zip(training_seqs, reward_seqs):
            error = np.zeros(5)
            delta_w = np.zeros(5)
            for t in range(len(seqs)):
                cur_state = seqs[t]
                cur_pred = np.dot(w, cur_state)
                error = lambd * error + cur_state
                if t != len(seqs) - 1:  # Not the terminal state
                    next_pred = np.dot(w, seqs[t + 1])
                    delta_w += alpha * (next_pred - cur_pred) * error
                else:  # Last state
                    delta_w += alpha * (rew - cur_pred) * error
            w += delta_w
        if np.linalg.norm(w_old - w) <= epsilon:
            break
    return w


# Calculate the average error for training sets
def td_training_sets_err(training_sets, reward_sets, lambd, w_true, alpha=0.01, epsilon=0.001):
    total_err = 0
    for training_seqs, reward_seqs in zip(training_sets, reward_sets):
        w_td = td_training_seqs(training_seqs, reward_seqs, lambd, alpha, epsilon)
        total_err += np.sqrt(np.mean((w_true - w_td) ** 2))
    return total_err / len(training_sets)


# Plot figure 3: Error vs Lambda
def figure3(lambd_values, w_true):
    avg_err_values = [td_training_sets_err(training_sets, reward_sets, lambd, w_true) for lambd in lambd_values]
    plt.plot(lambd_values, avg_err_values, marker='o')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.xlim((min(lambd_values) - 0.05, max(lambd_values) + 0.05))
    plt.ylim((min(avg_err_values) - 0.01, max(avg_err_values) + 0.01))
    plt.margins(x=0.5, y=0.15)
    plt.annotate('Widrow-Hoff', xy=(0.75, max(avg_err_values) - 0.01))
    plt.savefig('figure3')
    plt.show()


# Perform TD learning and update weights after each sequence
def td_training_seqs_update_seq(training_seqs, reward_seqs, lambd, alpha=0.01):
    w = np.ones(5) * 0.5
    for seqs, rew in zip(training_seqs, reward_seqs):
        error = np.zeros(5)
        delta_w = np.zeros(5)
        for t in range(len(seqs)):
            cur_state = seqs[t]
            cur_pred = np.dot(w, cur_state)
            error = lambd * error + cur_state
            if t != len(seqs) - 1:  # Not the terminal state
                next_pred = np.dot(w, seqs[t + 1])
                delta_w += alpha * (next_pred - cur_pred) * error
            else:  # Last state
                delta_w += alpha * (rew - cur_pred) * error
        w += delta_w
    return w


# Calculate average error with per-sequence updates
def td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha=0.01):
    total_err = 0
    for training_seqs, reward_seqs in zip(training_sets, reward_sets):
        w_td = td_training_seqs_update_seq(training_seqs, reward_seqs, lambd, alpha)
        total_err += np.sqrt(np.mean((w_true - w_td) ** 2))
    return total_err / len(training_sets)


# Plot figure 4: Error vs Alpha for different Lambdas
def figure4(lambd_values, alpha_values, w_true):
    for lambd in lambd_values:
        avg_err_values = [td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha) for alpha in
                          alpha_values]
        plt.plot(alpha_values, avg_err_values, marker='o', label=f'lambda = {lambd}')
    plt.legend()
    plt.xlim((min(alpha_values) - 0.05, max(alpha_values) + 0.05))
    plt.ylim((0.05, 0.75))
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.savefig('figure4')
    plt.show()


# Plot figure 5: Error with best Alpha for each Lambda
def figure5(lambd_values, alpha_values, w_true):
    best_alphas = {}
    for lambd in lambd_values:
        min_err, best_alpha = np.inf, 0
        for alpha in alpha_values:
            avg_err = td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha)
            if avg_err < min_err:
                best_alpha, min_err = alpha, avg_err
        best_alphas[lambd] = best_alpha

    avg_err_values = [td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, best_alphas[lambd]) for
                      lambd in lambd_values]
    plt.plot(lambd_values, avg_err_values, marker='o')
    plt.xlim((min(lambd_values) - 0.05, max(lambd_values) + 0.05))
    plt.ylim((min(avg_err_values) - 0.01, max(avg_err_values) + 0.01))
    plt.xlabel('Lambda')
    plt.ylabel('Error with Best Alpha')
    plt.annotate('Widrow-Hoff', xy=(0.75, max(avg_err_values) - 0.01))
    plt.savefig('figure5')
    plt.show()


# Generate training data
training_sets, reward_sets = generate_trainingsets(num_training_sets=100, num_seqs=10)

# Define constants for experiments
lambd_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
alpha_values = [0.05 * i for i in range(13)]
w_true = np.array([1, 2, 3, 4, 5]) / 6

# Plot results
figure3(lambd_values, w_true)
figure4(np.array([0, 0.3, 0.8, 1]), alpha_values, w_true)
figure5([i / 10 for i in range(11)], alpha_values, w_true)