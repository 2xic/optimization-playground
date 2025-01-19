
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

## [MatMul-Free LM](https://arxiv.org/pdf/2406.02528)
[Source code](https://github.com/ridgerchu/matmulfreellm)

- language model with no matmul operations.
- They use `BitLinear layers` which consistent of only {-1, 0, -1} for the weight values
  - This is the trick as then the operation can be represented as a addition or subtraction operation
  - They use Gated Linear Unit also
- Benchmarks show performance close to other methods, but with much more reduced memory usage.

## [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://openreview.net/pdf?id=AL1fq05o7H)
[MAMBA from Scratch: Neural Nets Better and Faster than Transformers](https://www.youtube.com/watch?app=desktop&v=N6Piou4oYx8)
- State space model
- RNN < Attention ?
  - RNN can't compute in parallel
  - RNN has vanishing gradients
  - [Linear RNN ?](https://openreview.net/pdf?id=M3Yd3QyRG4) 
    - Use Linear function instead of a neural network
    - SOTA of the Long Range Arena benchmark
    - Does not beat Transformers though ...
      - Problem: If values are close to 1 in previous layer this value will stay un-changed, if it is close to 0 then it will start to forget 
  - Mamba  
    - Use linear learned weights
    - 

## [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/pdf/2405.04517)
- LSTMs updated for the Transformer ERA (one of the co-author is also the original co-investor of LSTMs)
- Changes
  - (sLSTM) Exponential gates are added so that the layer can revise storage decisions.
  - (mLSTM) Instead of the memory cell being a scalar, it's now a matrix.
    - Key, value storage like the transformer :D 
- The results are on par with current transformer like models.

## [Were RNNs All We Needed?](https://arxiv.org/pdf/2410.01201)
- Make some modifications so that the RNNs don't need to backpropagate through time.
  - which makes them parallelizable
- Source code attached
- Results are comparable to other methods with Transformers
 
[Fchollet](https://x.com/fchollet/status/1841902521717293273)
- Deep learning tries to find a curve that describes the dataset.
- As you long as the dataset can fit on a curve - "all architectures will convergence to the same performance in the large-data paradigm"
-  
[Same as QRNN?](https://x.com/Smerity/status/1842012433281646606)
- at least close to.

### [SageAttention2](https://arxiv.org/pdf/2411.10958)
[Source code](https://github.com/thu-ml/SageAttention)
- Surpasses FlashAttention2
- 
