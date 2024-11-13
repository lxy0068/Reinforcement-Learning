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
### HW4: Reinforcement Learning - Taxi-v3 Environment

## **Overview**
This homework assignment involves implementing a Q-learning algorithm to solve a Markov Decision Process (MDP) for the Taxi-v3 environment. The task is to train an agent to pick up a passenger from one of four designated locations and drop them off at another, while maximizing rewards and minimizing penalties. The environment is deterministic, and the agent needs to learn an optimal policy using Q-learning.

### **Environment Description:**
- **Grid-based Environment**: The Taxi-v3 environment consists of a fixed grid where a taxi navigates between different locations to pick up and drop off passengers.
- **States**: The state space consists of different configurations of the taxi's location, passenger's location, and destination.
- **Actions**: The agent can take 6 actions:
  1. Move South
  2. Move North
  3. Move East
  4. Move West
  5. Pickup Passenger
  6. Drop-off Passenger
- **Rewards**:
  - Successfully dropping off a passenger at the correct location yields a reward of +20.
  - Incorrect attempts to drop off or pick up the passenger incur a penalty of -10.
  - Each time step costs the agent a reward of -1 to encourage efficiency.

### **Objective:**
Your task is to implement a Q-learning agent that:
- Trains in the Taxi-v3 environment.
- Learns the optimal Q-values for state-action pairs.
- Uses an epsilon-greedy policy to explore and exploit the environment.
- Returns the optimal Q-value for given state-action pairs.

---

## **Q-Learning Algorithm:**
Q-learning is a **model-free, off-policy** reinforcement learning algorithm. The agent learns a Q-value function $$\ Q(s, a) \$$ that estimates the expected future rewards of taking action $$\( a \)$$ in state $$\ s \$$. The update rule for Q-learning is:
$$\ 
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s, a) \right] 
\$$
Where:
- $$\ \alpha \$$ is the learning rate.
- $$\ \gamma \$$ is the discount factor.
- $$\ r \$$ is the reward received after taking action $$\ a \$$ from state $$\ s \$$.
- $$\ s' \$$ is the new state.

The agent balances exploration and exploitation using an **epsilon-greedy policy**:
- With probability $$\ \epsilon \$$, the agent takes a random action (exploration).
- With probability $$\ 1 - \epsilon \$$, the agent chooses the action with the highest Q-value (exploitation).

---

## **Instructions:**

### **Algorithm Details:**
- **Initialization**:
  - The Q-table is initialized to zeros for all state-action pairs.
  - The agent's epsilon (exploration rate) starts at 1.0 and decays over time to encourage exploration early in training and exploitation later.
  
- **Training**:
  - The agent interacts with the environment for a specified number of episodes.
  - For each episode, the agent:
    1. Resets the environment.
    2. Chooses an action using the epsilon-greedy policy.
    3. Updates the Q-value based on the observed reward and next state.
    4. Decays epsilon to reduce exploration as the agent learns more.

### **Tests**:
The implementation includes several unit tests using the `unittest` framework. These tests verify the correctness of your Q-learning agent by comparing the learned Q-values for specific state-action pairs to expected values.

---

## **How to Run:**
1. **Dependencies**:
   - Python 3.6.x
   - gym==0.17.2
   - numpy==1.18.0
   - unittest (for running tests)

2. **Run the notebook**:
   - Implement the `solve()` method to train the agent and populate the Q-table.
   - Use the `unittest` framework to run the test cases and validate the solution.

### **Environment Setup**:
Ensure you have the correct versions of the dependencies as listed above. If necessary, create a virtual environment and install the dependencies:

```bash
pip install gym==0.17.2 numpy==1.18.0
```

---

## **Resources:**
- **Chapters 6.5 and 2.6.1** of Sutton and Barto's *Reinforcement Learning: An Introduction* provide foundational understanding of Q-learning and off-policy temporal difference (TD) control methods.
- **Algorithms for Sequential Decision Making** by M. Littman discusses Q-learning and related algorithms for solving decision-making problems.

---

## **Notes:**
- The expected results should be accurate up to 3 decimal places, as specified in the test cases.
- The discount factor for this problem is set to $$\ \gamma = 0.9 \$$ to balance short-term and long-term rewards.
- The number of episodes for training is high to ensure the agent fully explores and converges to the optimal policy.

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

## Result

![figure3](https://github.com/user-attachments/assets/032d44f4-d8db-43d6-a3b8-e79db02c8b14)
![figure4](https://github.com/user-attachments/assets/d0764e45-159d-4993-92c9-c277fc808731)
![figure5](https://github.com/user-attachments/assets/15cdc8a0-55bd-4135-8450-d8702964c9ae)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

# Project 2: DQN for Lunar Lander

## Overview

This project focuses on training a Deep Q-Network (DQN) agent to solve the Lunar Lander problem using the OpenAI Gym's `LunarLander-v2` environment. The goal is to train the agent to land successfully while optimizing the reward over time. The project involves tuning key hyperparameters such as epsilon decay, gamma (discount factor), and learning rate to achieve optimal performance.

## Structure

1. **Code Implementation**: The code to train the DQN agent on the Lunar Lander environment, focusing on the impact of different hyperparameters.
   - `dqn_lunar_lander.py`: Main script to train the DQN agent.
   - `dqn_lunar_lander_Epsilon_Decay.py`: Script for training the agent with variations in the epsilon decay rate.
   - `dqn_lunar_lander_Gamma.py`: Script for training the agent with different values of the discount factor (gamma).
   - `dqn_lunar_lander_Learning_Rate.py`: Script for training the agent with varying learning rates.

2. **Graph Generation**: The code will produce graphs illustrating:
   - The reward per episode during training.
   - Performance over 100 consecutive episodes using the trained agent.
   - The effect of different hyperparameter settings on the agent's performance.

3. **Report**: A written report that details:
   - The problem being addressed.
   - The experiments conducted and their implementation.
   - Analysis of the results.
   - Comparisons between the performance of the agent with different hyperparameter settings.
   - Challenges encountered during training and how they were resolved.

## How to Run the Project

1. Install the required dependencies:
    ```bash
    pip install torch numpy matplotlib gym
    ```

2. Clone or download the project repository.

3. To train the DQN agent with default settings:
    ```bash
    python dqn_lunar_lander.py
    ```

4. To experiment with different hyperparameters, run the respective scripts:
    - Epsilon decay variation:
      ```bash
      python dqn_lunar_lander_Epsilon_Decay.py
      ```
    - Gamma variation:
      ```bash
      python dqn_lunar_lander_Gamma.py
      ```
    - Learning rate variation:
      ```bash
      python dqn_lunar_lander_Learning_Rate.py
      ```

5. The generated graphs will be saved as `.png` files in the `figures/` directory.

## Notes and Error Handling

- **Box2D Installation Issues**:
  - If you encounter errors related to Box2D when running `gym.make('LunarLander-v2')`, you may need to compile Box2D from source. Follow these steps:
    ```bash
    pip uninstall box2d-py
    git clone https://github.com/pybox2d/pybox2d
    cd pybox2d/
    python setup.py clean
    python setup.py build
    sudo python setup.py install
    ```
  - Windows users can use prebuilt Python wheels from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pybox2d).

- **Common Training Instabilities**:
  - **Diverging Rewards**: If the rewards start to diverge or become unstable, try reducing the learning rate in `dqn_lunar_lander_Learning_Rate.py`.
  - **Slow Convergence**: If the agent is learning too slowly, consider adjusting the gamma value in `dqn_lunar_lander_Gamma.py` to focus more on immediate rewards or fine-tune the epsilon decay for better exploration-exploitation balance.
  - **Exploding Gradients**: If you encounter exploding gradients during training, use gradient clipping in PyTorch:
    ```python
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ```

- **Runtime Errors**:
  - If you encounter memory-related issues, try reducing the batch size or clearing the replay buffer periodically:
    ```python
    if len(replay_buffer) > buffer_size:
        replay_buffer.pop(0)
    ```

## Result

![gameplay](https://github.com/user-attachments/assets/a2547be3-7828-400d-a1eb-88143defedb7)
![Figure_1](https://github.com/user-attachments/assets/ba0a1b24-2b39-45a7-b2e0-2e8d33b135c0)
![Figure_2](https://github.com/user-attachments/assets/f0ed4f3b-9d4a-4bb0-b7aa-e4fcd235181c)
![Figure_3](https://github.com/user-attachments/assets/600fbe3b-649f-40cb-829d-4d120c5a21da)
![Figure_4](https://github.com/user-attachments/assets/d112f530-aad8-4a03-872b-eb7a650d882e)
![Figure_5](https://github.com/user-attachments/assets/6216cad2-074c-4f3e-bced-7b0d3879ecc2)
![Figure_6](https://github.com/user-attachments/assets/0e3ff07e-c5c0-43d6-a625-713459902d0a)
![Figure_7](https://github.com/user-attachments/assets/5aa0896b-c930-442e-a52f-001e626c80f6)
![LunarLander-v2](https://github.com/user-attachments/assets/8da0c3a2-e8c9-4d7f-89b1-553cf8e60e98)
![result](https://github.com/user-attachments/assets/32883bd4-ff37-4e87-a834-51bd3f03a91e)

## Resources

- **OpenAI Gym**: [Lunar Lander Environment](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py).
- **Reinforcement Learning Texts**: Sutton & Barto, *Reinforcement Learning: An Introduction*.
- **Lectures**: Reinforcement Learning (Lesson 8: Generalization).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
