# [Neural Episodic Control](https://arxiv.org/pdf/1703.01988.pdf)

`Method that attempts to be more sample efficient by giving the agent a buffer of past experiences.`

Why are current RL solutions not sample efficient ? 
1. SGD requirers a lower learning rate for a neural network to converge
2. Sparse rewards in the environments
3. Slow reward propagation because we train on mini batches 

The Neural dictionary allows lookups and writes, and is also used for the Q function (as it's a classical table lookup). Any action taken will be written into this "memory bank". This allows for more efficient trancing, and is the main idea from the paper-

The results look good on the plot, so I kinda want to implement this paper, and the idea is cool. 

TODO: implement ? 

