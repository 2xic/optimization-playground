## [spreading vectors for similarity search]((https://arxiv.org/pdf/1806.03198.pdf))
[code](https://github.com/facebookresearch/spreadingvectors)

- Usually state of the art indexers will train a a quantifiers on the training data. i.e quantizer is adapted to the data.
- This paper flips that around and adapt the data to the quantizer
  - Last layer of the neural network forms a fixed parameter-free quantizer
  - One objective used is to try to keep the spherical latent space and 
- Building upon the idea of the [learned indexes](https://arxiv.org/abs/1712.01208) - or taken inspiration from might be a more correct word for it.
- Training
   -   [maximum entropy ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7539664)
       -   Defined in section 3.1 quite straightforward
    -  [standard ranking loss for preserving neighbors are preserved](https://gombru.github.io/2019/04/03/ranking_loss/)
       -  Defined in section 3.2 quite straightforward
    -  Total loss is the sum of the two losses above
 -  The model architecture itself is quite simple
    -  3 layers with ReLu


It was also mentioned in this [High-D Vector Similarity Search talk](https://vldb.org/2021/files/slides/tutorial/tutorial5.pdf)
