### [The Forward-Forward Algorithm: Some Preliminary Investigations](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

- Replaces forward and backward passes with two forward passes
  - One with real data
  - One with fake data
- Each layer has it's own objective function    
  - Based on ideas
    - https://arxiv.org/abs/1809.01812
    - https://www.cs.toronto.edu/~fritz/absps/cogscibm.pdf
  - Which is simply to have high score when positive data, and low score with negative data
    - Objective function could i.e sum of squared activities in the layer
    - P(positive) = sigma(
        sum(y_j ** 2 - theta) 
      )
      - Theta is threshold
      - y_j is activity of hidden unit
      - sigma = logistic function
        - i.e sigmoid
  -  " FF normalizes the length of the hidden vector before using it as input to the next layer"
  -  

**Update from 2023** There is also [this](https://medium.com/@Mosbeh_Barhoumi/forward-forward-algorithm-ac24d0d9ffd) medium post which is quite nice. 
