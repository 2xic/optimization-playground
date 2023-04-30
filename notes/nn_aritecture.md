
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
