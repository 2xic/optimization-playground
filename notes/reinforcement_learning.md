## Reinforcement learning
For the paper notes that are so short that they don't need their own file.

### [Equivariant MuZero](https://arxiv.org/pdf/2302.04798.pdf)
Tries to make MuZero more data efficient by having symmetry be apart of the world model.

# [RL^2 - fast reinforcement learning via slow reinforcement learning](https://arxiv.org/pdf/1611.02779.pdf)
Idea is to learn a reinforcement learning within RNN weights using a (existing) RL method. In other words (and I quote the paper) learning a RL algorithm as a reinforcement learning problem 

-> Cool idea -> TODO: Implement ? 

## [Go-explore: a new approach for hard-exploration problems](https://arxiv.org/pdf/1901.10995.pdf)
Core idea: Explore new states by revisiting old promising ones

1. Remember state that are previously visited
2. Return to a promising state (no exploration)
3. Start exploring
4. Run imitation learning on the best trajectory 

There are more to the idea, but that is the rough idea.
This method was enough to beat the record for "Montezuma’s Revenge" by 4x the previous state of the art.
