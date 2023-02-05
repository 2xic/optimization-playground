**Model is still not converging, tried various things, but none of it gave any good results.**
- Changing various aspects of the loss
- Using deeper model like resnet as mentioned in the paper
  - Also tried changing out the activations, and kernel size
- Switching up learning rate
  - This seems to give unstable results, even though this is what the authors of the paper did
  - The result is just that all outputs converge to the same thing
- Tried to look at the outputs of the encoder and see if it has just converged to something resembling a 0 vector, this does not seem to be the case
  - The output all have various patterns currently, but still does not help when doing transfer learning. 
  - Also visualized this, and tested that is does not output the same thing


**Bugs found** 
- Bug in how the weights schedule was set, error in the cosine function
-  Would have been found earlier if I just wrote a unit test to begin with
- 

**Things to try out**
- ~~Maybe the bug is how the feature learning is done ? Try to check the encoder model output on various classes. and see how dissimilar they are~~
  - ~~I just find it super weird the fact that the model is not able to give anything reasonable.~~
- 
