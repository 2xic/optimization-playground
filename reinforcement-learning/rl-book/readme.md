# Rl-book
Reproducing figure and algorithms from [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

### Chapter 2 - Multi-armed bandits
[Figure 2.2 - Average reward](./multi-armed-bandits/simple_average_reward.png)

[Figure 2.2 - Average optimal action](./multi-armed-bandits/simple_average_reward.png)

[Figure 2.3 - Optimistic initial value ](./multi-armed-bandits/optimistic_initial_value_average_optimal_action.png)

[Figure 2.4 - UCB vs e-greedy ](./multi-armed-bandits/eps_vs_ucb_average_reward.png)

[Figure 2.5 - Gradient bandit](./multi-armed-bandits/gradient_bandit_average_optimal_action.png)

[Figure 2.6 - Parameter study](./multi-armed-bandits/parameter_study.png)
- **(does not match the book plot well currently)**

### Chapter 4 - Dynamic programming
Skipped (nothing that special there)

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

[Figure on page 132 - Q-Learning vs Sarsa](./temporal-difference-learning/Q_learning_Sarsa_cliff_walking.png)

(todo) Figure 6.3 - TD-Control performance

[Figure 6.5 - Comparison of Q-Learning and Double Q-Learning](./temporal-difference-learning/bias_q_learning_vs_double_q_learning.png)

[Algorithm - Double Q-Learning for estimating Q](./temporal-difference-learning/double_q_learning.py)

### Chapter 7 - n-step bootstrapping
[Algorithm - n-step TD for estimating value function](./n-step-Bootstrapping/n_step_td_for_estimating_v.py)

(todo) - Figure 7.2 - Performance of n-step TD methods as a function of alpha, for various values of n, on a 19-state random walk task.

[Algorithm - n-step Sarsa for estimating Q function](n-step-Bootstrapping/n_step_sarsa_for_q_function.py)

(todo) - Figure 7.4 - Visualize the difference in action between one-step Sarsa and 10 step Sarsa

(todo) Algorithm - Off-policy n-step Sarsa for estimating Q function

(todo) Algorithm - n-step Tree backup for estimating Q function

(todo) Algorithm - Off-policy n-step Q(sigma) for estimating Q

## Chapter 8 - Planning and Learning with Tabular Methods
[(Unofficial) Slides on this chapter](https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/16-17-planning-and-learning.pdf)

[Algorithm - Random-sample one-step tabular Q-planning](./planning-and-learning-wth-tabluar-methods/random_sample_one_step_tabluar_q_planning.py)
- [Plot of the reward against random agent](./planning-and-learning-wth-tabluar-methods/plot/random_sample_with_one_step_tabluar_q_planning.png)

(todo) Algorithm - Tabluar Dyna-Q

(todo) Algorithm - Prioritized sweeping for a deterministic environment

### Chapter 13 - policy gradient methods
- Learns a parametrized policy
- Can use a value function, but does not need to.
  
[Algorithm - REINFORCE: Monte-Carlo Policy-Gradient Control (episodic)](./policy-gradient/reinforce.py)
- [Plot of the reward against random agent](./policy-gradient/plot/reinforce.png)

[Algorithm - REINFORCE with baseline](./policy-gradient/reinforce_with_baseline.py)
- [Plot of the reward against random agent](./policy-gradient/plot/reinforce_with_baseline.png)

[One-step Actor–Critic (episodic)](./policy-gradient/one_step_actor_critic.py)
- [Plot of the reward against random agent](./policy-gradient/plot/one_step_actor_critic.png)

