
## Notes from the paper
[Paper](https://arxiv.org/pdf/1911.08265.pdf)

- Tree based search + learned model
- Learns the reward, action policy, and value function

### Planning, acting, and training with a learned model

How the model sees the world
- The model has 3 components : representation, dynamics, and predictor
- representation function takes input (the initial board state) -> (hidden state) 
  - Hidden state is the model representation of the game
- dynamic functions takes input (previous hidden state, action) -> (reward, new hidden state)
- prediction function takes input (hidden state) -> (policy, value)

How the model act in the world
- Monte carlo tree search is applied based on how the model sees the world (see Figure 1)
- The action sampled and observation + reward is added into a replay buffer

How the model learns the world
- Trajectory is sampled from the replay buffer
- For each observation -> run the representation function 
  - The model is then unrolled for K steps 
    - dynamic function receives (previous hidden state, real action)  where "real action" means the one added to the replay buffer
  - The goal is to train the model as follows
    - policy should mimic the tree search 
    - value function should be n-step reward
    - dynamic function should predict the reward.
  
#### The algorithm - deeper dive
- Representation should encode all previous observations (?)
  - Why ? Makes sense for games, but not boardgames, I think
-  All the other things are covered by figure 1, and above comment, I think


#### Some things worth mentioning
- MuZero does not simulate the environment, it's only the hidden state that is used during the search.
- MuZero is only masked for legal actions at the root. 
  - Any nodes further down is not masked, and can do "any" action
- MuZero does not "know" what a terminal node is, it will continue to search.

- They use a custom function to give a action score (Appendix B) 
  - Same is true for the backup


- Equation 6 layout the policy function for the tree
- Appendix G might be worth looking at once more also.
