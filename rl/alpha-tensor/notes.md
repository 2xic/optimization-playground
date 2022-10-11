

## Notes from the paper
[Paper](https://www.nature.com/articles/s41586-022-05172-4)

- "Agent trained to play a single-player game where the objective is finding a tensor decompositions within a finite factor space"
- It's able to find algorithms more effective on various sizes, and more relevant 4x4 where it sets the new record, and that hasn't been done in 50 years as far as the authors are aware.
  
- Trained end2end to recognize and generalize to predict efficient decompositions.

### The game ("TensorGame")
- The player select show to combine different entires of the matrices to multiply.
- Score is set based on number of selected operations required to reach the correct multiplication results
  - Huge action space!
- Same idea as AlphaZero
  - Neural network is trained to guide the planning
  

- S_0 = Target tensor we want to decompose
- At each timestep the agent has to select an action (u, v, w)
- The S_t tensor is then updated by subtracting the rank tensor by
  - S_t < - S_t - u*v*w
- Goal of the game is for the player to reach the zero tensor, but with the smallest amount of moves
  - An limit is in place to terminate the game if it goes on for 2 long   
- Reward is -1 for each action taken
  - An additional reward gamma * (S_R_limit) is also given is the R_limit is hit
- 

### Still monte carlo
- AlphaTensor also uses monte carlo tree search to guide the model
- 

### Architecture
- Transformer-based
- Synthetic demonstrations is used to "warm up" the network

----

The game, and layout is described in more details on page 8 to 10.


---

# action ? 
We should code a version of AlphaZero, and then maybe tune it for this task. 
