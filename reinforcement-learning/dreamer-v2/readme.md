### [MASTERING ATARI WITH DISCRETE WORLD MODELS](https://arxiv.org/pdf/2010.02193.pdf)

Initial thought was that it reminded me of MuZero, but it is not actually that similar.

## Notes
Some key points
- Model is trained jointly with a component for representation, transition, image, reward, discounter. There is also a recurrent model. 

## World model
- representation model (part of the core model)
  - takes in the env image and the hidden state
  - outputs a latent state
- transition predictor (part of the core model)
  - tries to predict the current image without access to it (?)
  - it does it with only the recurrent layer
- Recurrent model (part of the core model)
  - Takes the last hidden state, representation and action
  - Outputs new hidden state (used in the other components)
-------
- image predictor
  - Input hidden state and image representation
  - *Used to train the representation* 
  - ^ they use a discrete space (one hot encoder, not floats)

## Actor critic
Works on the representation, not the actual env image. 

- reward predictor
  - Input hidden state and image representation
- discounter predictor
  - Input hidden state and image representation

The actor is actually trained to output the actions. The critic should output the value of the state. 

## Searching
- I think what is done is that the sequence model is used to search over all actions to create the search tree

## Loss
- Predicted image reconstruction
- Predicted reward and discounter
- Predicted 

## Tricks they use  (v3)
- categorical variables 
  - https://pytorch.org/docs/stable/distributions.html#onehotcategorical
- symlog


## Other links
- https://xlnwel.github.io/blog/reinforcement%20learning/DreamerV2/
- https://arxiv.org/pdf/1912.01603.pdf which is the v1 of the paper
- https://arxiv.org/pdf/2301.04104v1.pdf which is the v3 of the paper. 
- https://www.youtube.com/watch?v=vfpZu0R1s1Y v3 overview of the paper
- https://www.youtube.com/watch?v=o75ybZ-6Uu8 v2 overview of the paper 
- https://danijar.com/project/dreamerv2/
- https://www.youtube.com/watch?v=_e3BKzK6xD0 <- quick video from the authors, also good
- 

