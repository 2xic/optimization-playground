# Rl-book
Reproducing figure and algorithms from [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

### Chapter 2 - Multi-armed bandits
[Figure 2.2 - Average reward](./multi-armed-bandits/simple_average_reward.png)

[Figure 2.2 - Average optimal action](./multi-armed-bandits/simple_average_reward.png)

[Figure 2.3 - Optimistic initial value ](./multi-armed-bandits/optimistic_initial_value_average_optimal_action.png)

[Figure 2.4 - UCB vs e-greedy ](./multi-armed-bandits/eps_vs_ucb_average_reward.png)

[Figure 2.5 - Gradient bandit](./multi-armed-bandits/gradient_bandit_average_optimal_action.png)

[Figure 2.6 - Parameter study (does not match the book plot well currently)](./multi-armed-bandits/parameter_study.png)

### Chapter 5 - Monte carlo methods
(todo) Algorithm - First visit MC prediction, for estimating Value function 

(todo) Algorithm - Monte carlo ES (Exploring starts), for estimating policy

(todo) Algorithm - On-policy first-visit MC control (for e-soft policies), estimates policy

(todo) Figure 5.3  - Weighted importance

(todo) Figure 5.4  - Ordinary importance

(todo) Algorithm - Off-policy MC prediction (policy evaluation) for estimating Q

[Algorithm - Off-policy MC control, for estimating policy](./monte_carlo_methods/off_policy_mc_control.py)

### Chapter 6 - Temporal-Difference learning
[Algorithm - Tabular TD (0) for estimating value_policy](./temporal-difference-learning/tabluar_td_0.py)

[Algorithm - Sarsa (On-policy TD control) for estimating](./temporal-difference-learning/sarsa.py)

[Algorithm - Q-Learning (Off-policy TD-control) for estimating policy](./temporal-difference-learning/q_learning.py)

[Figure on page 132 - Q-Learning vs Sarsa (does not match the book plot well currently)](./temporal-difference-learning/Q_learning_Sarsa_cliff_walking.png)

(todo) Figure 6.3 - TD-Control performance

(todo) Figure 6.5 - Comparison of Q-Learning and Double Q-Learning

[Algorithm - Double Q-Learning for estimating Q](./temporal-difference-learning/double_q_learning.py)

