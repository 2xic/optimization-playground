
## [Algorithmic progress in computer vision](https://arxiv.org/pdf/2212.05153.pdf)
Is recent progress in machine learning a results of more compute or from or algorithms and architectures ? If we believe the authors, both have contributed. 
This works seems to be partially inspired by [Deep Neural Nets: 33 years ago and 33 years from now](http://karpathy.github.io/2022/03/14/lecun1989/) where Karapthy gradually applied modern NN techniques to an old NN and see what happens. Other research has also shown that modern NN techniques on old base models show that compute is not everything.
-> However most algorithmic progress is ahe results of improved compute

## [Deep Neural Nets: 33 years ago and 33 years from now](http://karpathy.github.io/2022/03/14/lecun1989/)
Karpathy switches brings new life to an old model architecture
-> The model was trained originally trained with MSE regression  This was replaced with Cross entropy.
-> SGD is replaced with Adam with weight decay
-> Adding some data augmentation (and also increasing the epochs to compensate) which gave a nice boost
-> Added dropout and switched out activation functions which also gave a nice boost (Karpathy mentions that the improvement boost is mostly from dropout)

Just scaling up the dataset gives quite the improvement also :) 

## [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.pdf)
Using batch normalization makes it easier to train neural networks without thinking about the learning rate. Why ? Because the distribution between layers will be more normalized. Since the input passes through all layers during training this makes the distribution much less noisy. 

## [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113)
[Found on twitter](https://twitter.com/ibab_ml/status/1669579636563656705)

The most interesting results from this paper imo is the fact that you cna tradeoff the search time of the search and the training time. For each additional 10x of train-time compute, 15x of test-time compute (search time) can be removed.

