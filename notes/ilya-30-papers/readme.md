## Ilya Sutskever recommended papers


There are a few places that list theses
- [Primers • Ilya Sutskever's Top 30](https://aman.ai/primers/ai/top-30-papers/#gpipe-easy-scaling-with-micro-batch-pipeline-parallelism)
- [The paper list](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)

```
If you really learn all of these, you’ll know 90% of what matters today
```
(quote from Ilya _allegedly_)

There are some blog posts which I won't dive to deep into, but here they are for context-
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) whcih I read part of the transformers [implementation](../../nlp/transformers/)
- [The First Law of Complexodynamics](https://scottaaronson.blog/?p=762) which talks about [this idea](https://149663533.v2.pressablecdn.com/complexity-small.jpg). Complexity reaches a peak before turning around, but the disorder will keep increasing.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) which in many ways got a new renaissance after the world got hooked on Transformers. Maybe what I like the most here is the visluazation towards the end.
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) which I should take a more deep dive into with [memory cells](../concepts/memory_cells.md)


## [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/pdf/1811.06965)
This is built [into torch](https://pytorch.org/docs/stable/distributed.pipelining.html). Smaller batch sizes to allow more work to be done at the same time and all updates are synchronized at the end. [Lilian Weng](https://lilianweng.github.io/posts/2021-09-25-train-large/) also has some good blog post on this.

## [Recurrent neural network regularization](https://arxiv.org/pdf/1409.2329)
[Source code](https://github.com/wojzaremba/lstm)

Dropout doesn't really work for RNNs and LSTMs. Unless you do it for the non recurrent components which is what they propose in this paper.

## [Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)
[Minimum description length](https://en.wikipedia.org/wiki/Minimum_description_length)

- Weight pruning (dropout less formalized ?)
- Uses a coding schema to encode the weights

TODO: need to look more into this one.

## [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
Did some reant on this paper in [is ml driven by engineers or reasearch](../rants/is_ml_driven_by_engineer_or_reserach_improvment.md). They key insight from the paper was that deep networks + compute allowed neural networks to actually work in practice. Evreyone switched to neural networks after that.
[Funny thing is that it's almost eqvivalent to Lenet made more than a decade before, just deeper.](https://en.wikipedia.org/wiki/LeNet#/media/File:Comparison_image_neural_networks.svg)

## [Pointer Networks](https://arxiv.org/pdf/1506.03134)
[Papers with code](https://paperswithcode.com/method/pointer-net)

[Blog post TSP from dynamic programming to deep learning](https://myencyclopedia.github.io/en/2020/tsp-3-pointer-net/)

It's a kind of sequence of sequence model where the output points back to the original input sequence. The application was for Traveling sales man problem (TSP) and Convex hulls. I'm mostly intrested in TSP so focusing on that part of the paper.
The [TSP table](https://arxiv.org/pdf/1506.03134#page=8) show that the model doesn't divergece much form the other algorithms except for the last row. The authors also mentioned that the model will learn to generalize even if it's trained on the worst algortihm in other words, train the model on the output of A1 and it will make the model beat A1. 


## [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
See [resnet](../../computer-vision/resnet/), the idea is to have the input of the residucal block be part of the output.
```
X = Input to the block
F(X)  = The core residual block
```

Output is then `relu(F(X) + X)`

## [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
See [Transformers](../../nlp/transformers/)

## [Neural Turing Machines](https://arxiv.org/pdf/1410.5401)
See [Token turning machines](../../architecture/token_turing_machines.md), but I should explore this more also.

## [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)
I love scaling laws, see [scale](../scale.md) and [scale folder](../model-scale/).

## [Order Matters: Sequence to sequence for sets](https://arxiv.org/pdf/1511.06391)
- todo

## [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
- todo

## [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212)
- todo

## [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- todo

## [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027)
- todo

## [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427)
TLDR of the paper 
- $$RN (O) = f_{\omega} ( \sum_{i,j} g_{\Omega} (o_i, o_j))$$
- Where $O$ are set of objects (`i` and `j` are the nth objects), f and g are two neural networks.
- See [Figure 2](https://arxiv.org/pdf/1706.01427#page=6) for how this would work in practice.


Other sources
- https://harshakokel.com/posts/relational-network/
- https://ameroyer.github.io/architectures/a_simple_neural_network_module_for_relational_reasoning/

## [Variational Lossy Autoencoder](https://arxiv.org/pdf/1611.02731)
- Paper from OpenAi
- References Bit-back coding a lot, there is a good blogpost on that [here](https://deeprender.ai/blog/BitsBackCoding).
- Some general paper notes
  - VAE focuses on representation learning, VLAE focuses on global structure. This improves the output after the reconstruction.
  - https://www.cs.ubc.ca/labs/lci/mlrg/slides/2020-03-17-MLRG-VLAE-Dylan.pdf
  - https://ruishu.io/2017/01/14/one-bit/
  - https://github.com/duanzhiihao/lossy-vae/tree/main
  - https://encode.su/threads/3886-Very-simplifued-small-example-Bits-back-coding
  - http://zhat.io/bits-back-coding/
  - 

## [Relational recurrent neural networks](https://arxiv.org/pdf/1806.01822)
- todo

## [Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton](https://arxiv.org/pdf/1405.6903)
- todo

## [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595)
- todo

## [A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/pdf/math/0406077)
- todo

## [Machine Super Intelligence](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf)
Shane Legg (co-founder of DeepMind)
- [slides](https://pdfs.semanticscholar.org/e758/b579456545f8691bbadaf26bcd3b536c7172.pdf) 
- 

## Kolmogorov Complexity and Algorithmic Randomness
[Book](https://www.amazon.com/Kolmogorov-Complexity-Algorithmic-Randomness-Mathematical/dp/1470431823/ref=sr_1_1?crid=VGMWSMEXUNHK&dib=eyJ2IjoiMSJ9.MaOBOm3zOhbUF6HPcTrBlr9Cdwj_FBFyiowKOoXHcLgJuwUfdCiWc7w5Dg5PQKpzf0abKBShqG6SRwy0XW5x4nAhn1ks7Om6PqFBrQVH2X2AKeVQSNiqzrtg_jMZEKnLTf6t4GSwcfiQyvYQhYDvCg.r7m6lufVMX5tGVepNIsyPyPN5LkkgdRGByodt1aH_s8&dib_tag=se&keywords=Kolmogorov+Complexity+and+Algorithmic+Randomness&qid=1729951561&sprefix=%2Caps%2C236&sr=8-1)

## cs231n
[Standford course](https://cs231n.github.io/)
