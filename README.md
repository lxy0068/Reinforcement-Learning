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

   - Implement the `solve()` method to compute the λ-return, $$\G_t^\lambda\$$, for a given Markov reward process.
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
