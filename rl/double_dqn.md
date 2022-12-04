[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)

- Q-Learning is known to overestimate action values under certain conditions
- Douce Dqn reduces overestimation, and results in improved performance

----

Problem with Q-learning is that the same Q function is used both to select and to evaluate an action. 

To prevent this, we can decouple the selection from the evaluation.

So we use two functions, and swap the parameters (see equation 4)

----
Theorem 1 shows something intreating, not much error is needed error needed to affect the outcome

- TODO -> Replicate the plot ?

---



