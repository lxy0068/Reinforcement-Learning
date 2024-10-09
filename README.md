# HW1: Reinforcement Learning - DieN MDP Game

## Overview

This homework assignment involves solving a Markov Decision Process (MDP) for a dice-based game called "DieN." The objective is to compute the expected value of the winnings under an optimal policy. The game rules and dynamics are modeled as an MDP, and solving this involves determining the optimal state-value function.

## Instructions

1. **Game Description**:
   - The game consists of rolling a die with several possible outcomes.
   - On each turn, the player has two choices:
     - **Roll**: If the player rolls a good side, they accumulate money and can continue playing.
     - **Quit**: The player can stop playing and keep the accumulated winnings.
   - Rolling a bad side results in losing all the money, ending the game.
   
2. **Objective**:
   - Implement the `solve()` method to determine the expected winnings for a given configuration of good and bad sides on the die.
   - The method should return the expected number of dollars with an optimal policy.

3. **MDP Solution**:
   - Model the problem as an MDP with:
     - States representing the current bankroll.
     - Actions for rolling the die or quitting the game.
     - Rewards based on winning or losing the game.
   - The optimal policy should maximize the total expected winnings.

4. **Tests**:
   - The notebook includes test cases using the `unittest` framework.
   - To verify the implementation, run the notebook to check if your solution passes the provided test cases.

## How to Run

1. Ensure you have the necessary dependencies:
   - Python 3.x
   - NumPy
   - `unittest` (for running tests)

2. Open the Jupyter notebook and follow the steps to implement the `solve()` method.

3. Run the notebook to validate your solution with the provided test cases.

## Resources

- Chapters 3.6 and 4.3-4.4 from *Sutton and Barto's Reinforcement Learning* ([Link](http://incompleteideas.net/book/the-book-2nd.html)).
- *Algorithms for Sequential Decision Making* by M. Littman (Chapters 1-2).

## Notes

- The expected result should be correct up to 3 decimal places (e.g., 3.141).
- The discount factor is assumed to be `γ = 1`.

---

# HW2: Reinforcement Learning - TD(λ) Algorithm and λ-Return

## Overview

This homework assignment involves computing the λ-return and understanding its relationship with the Temporal Difference (TD) learning method, specifically TD(λ). The task is to apply these concepts to a Markov reward process.

## Instructions

1. **Objective**:

   - Implement the `solve()` method to compute the λ-return, $$G_t^\lambda$$, for a given Markov reward process.
   - The λ-return is a weighted combination of the $$\ n \$$-step returns, which is used in the update rule for the TD(λ) algorithm. It is given by:

$$ 
G_t^\lambda = \sum_{n=1}^\infty (1-\lambda)\lambda^{n-1} G_{t:t+n} 
$$ 


   where $$\ G_{t:t+n} \$$ is the $$\ n \$$-step return.

3. **Markov Reward Process**:
   - The assignment involves working with a Markov reward process, which can be modeled as a Markov Decision Process (MDP) with a single action possible from each state.
   - Given a set of states, rewards, and a probability transition function, the goal is to calculate the λ-return based on the provided discount rate `γ = 1`.

4. **TD(λ) Algorithm**:
   - The λ-return can be viewed as the target for the TD(λ) prediction algorithm.
   - You will implement the λ-return formula and compute the return for a given state and time step.

5. **Tests**:
   - The notebook includes test cases implemented with the `unittest` framework.
   - After implementing the `solve()` method, run the notebook to ensure that your solution passes the provided test cases.

## How to Run

1. Ensure you have the necessary dependencies:
   - Python 3.x
   - NumPy
   - `unittest` (for running tests)

2. Open the Jupyter notebook and implement the `solve()` method.

3. Run the notebook to validate your solution with the provided test cases.

## Resources

- Chapters related to TD(λ) from *Sutton and Barto's Reinforcement Learning* ([Link](http://incompleteideas.net/book/the-book-2nd.html)).

## Notes

- The expected result for each test case should be accurate up to 3 decimal places.
- The task assumes a discount factor of `γ = 1`.

---
# HW3: Reinforcement Learning - SARSA Agent

## Overview

This project involves building a **SARSA agent** to learn policies in the OpenAI Gym's **Frozen Lake** environment. The goal is for the agent to navigate the frozen lake grid to reach the goal (`G`), avoiding holes (`H`), while learning the optimal policy using the SARSA (State-Action-Reward-State-Action) algorithm.

## Requirements

- Python 3.x (developed using Python 3.11.7)
- **OpenAI Gym** for the Frozen Lake environment
- **NumPy** for numerical operations
- **Unittest** for testing the SARSA agent

## Environment Description

The Frozen Lake environment is a 4x4 or 8x8 grid where:

- `S`: Start point
- `F`: Frozen surface, where the agent can walk
- `H`: Holes, which the agent must avoid
- `G`: Goal, which the agent must reach

The agent receives a reward of `1` for reaching the goal and `0` for any other action.

## Project Structure

1. **FrozenLakeAgent Class**: Implements the SARSA learning algorithm to solve the Frozen Lake problem. The agent learns from the stochastic environment using the following key parameters:
   - **Gamma (γ)**: Discount factor for future rewards.
   - **Alpha (α)**: Learning rate.
   - **Epsilon (ε)**: Exploration rate for the epsilon-greedy policy.
   - **N_episodes**: The number of training episodes.

2. **SARSA Algorithm**: The agent updates its policy based on the SARSA update rule, which allows learning from interaction with the environment.

3. **Test Cases**: Unittests are used to validate the agent's learning and policy correctness in different scenarios using pre-defined parameters.

## How to Run the Project

1. Install the required dependencies:

   ```bash
   pip install gym numpy
   ```

2. Run the notebook or use the provided `FrozenLakeAgent` class to train the agent:

   ```python
   from frozen_lake_agent import FrozenLakeAgent
   
   agent = FrozenLakeAgent()
   optimal_policy = agent.solve(
       amap='SFFFHFFFFFFFFFFG',
       gamma=1.0,
       alpha=0.25,
       epsilon=0.29,
       n_episodes=14697,
       seed=741684
   )
   print(optimal_policy)
   ```

3. Run the unittests to validate the SARSA agent's performance:

   ```bash
   python -m unittest discover -s tests
   ```

## Testing

The project contains five predefined test cases, each checking the agent's ability to learn optimal policies under different configurations of the Frozen Lake environment. These test cases ensure that the SARSA agent behaves correctly across various scenarios, including grid size and hyperparameter values.

---

# Project 1: TD(λ) Replication

## Overview

This project involves replicating the experiments presented in Richard Sutton's 1988 paper "Learning to Predict by the Methods of Temporal Differences." Specifically, the goal is to recreate the results from Figures 3, 4, and 5, and compare these results to those found in Sutton’s textbook, "Reinforcement Learning: An Introduction" (Chapter 7 and Chapter 12).

The project includes a written report detailing the experiment, implementation, outcomes, analysis, and any challenges encountered during the replication process.

## Requirements

To run the code and reproduce the experiments, you will need the following:

- **Python 3.x**
- **Matplotlib** for generating graphs
- **NumPy** for numerical computations

## Structure

1. **Code Implementation**: The code necessary to replicate the TD(λ) experiments from the paper, specifically focusing on Figures 3, 4, and 5.
2. **Graph Generation**: The code will produce graphs that replicate the figures from the original paper, illustrating how the algorithm performs under different parameter settings.
3. **Report**: A written report that details:
   - The problem being addressed
   - The experiments conducted and their implementation
   - A thorough analysis of the results
   - Comparisons between the replicated results and the original paper’s results
   - Any issues or pitfalls encountered during replication, and how they were resolved

## How to Run the Project

1. Install the required dependencies:
    ```bash
    pip install numpy matplotlib
    ```

2. Clone or download the project repository.

3. To generate the results:
    ```bash
    python test.py
    ```

4. The graphs corresponding to Figures 3, 4, and 5 will be saved as `.png` files.

## Project Files

- `replicate_td_lambda.py`: Main script to replicate the experiments and generate the graphs.
- `README.md`: This file, providing instructions and an overview of the project.
- `report.pdf`: A detailed report analyzing the replication process, results, and challenges.
- `figures/`: Folder containing the generated figures for comparison with the original results.

## Analysis

The report includes a discussion of the following:

- **How the replicated results match (or differ from) the original results**:
  - Differences in performance and possible reasons (e.g., different parameter settings or environmental variations).
  - Comparisons between the replicated results and those found in Chapter 7 of Sutton's textbook.

- **Challenges**:
  - Any unclear parameters or ambiguities in the original paper.
  - Steps taken to overcome issues encountered during replication.
  - Assumptions made in the replication process and their justifications.

## Resources

- **Richard Sutton's Paper**: "Learning to Predict by the Method of Temporal Differences" (1988).
- **Reinforcement Learning: An Introduction**: Sutton & Barto (2nd Edition, Chapters 7 & 12).
- **Lectures**: Reinforcement Learning (Lesson 3: Temporal Difference Learning).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
