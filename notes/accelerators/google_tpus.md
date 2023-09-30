## Google TPUs
Supposedly based on this [lecture from Taylor University](https://www.youtube.com/watch?v=9Mo80a4s0Bs), Google started to rent out the TPUs because they had this extra compute that was not utilized always.

### [OpenTPU](https://github.com/UCSBarchlab/OpenTPU)
Theses have tried to reimplement the Google TPUs based on  [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)

### [In-Datacenter Performance Analysis (Presentation)](https://www.cse.wustl.edu/~roger/566S.s21/In-Datacenter%20Performance%20Analysis%20of%20a%20Tensor%20Processing%20Unit.pdf)
Looking at this before the paper, and quite infesting. They wanted to make interference 10x that of GPU in about 15 months. It's also interesting that 3 types of NNs represent 95 % of Google interference workload (this has probably changed now).

### [In-Datacenter Performance Analysis of a Tensor Processing Unit (Paper)](https://arxiv.org/abs/1704.04760)
*TODO*

### [Domain-Specific Architectures for Deep Neural Networks](https://inst.eecs.berkeley.edu//~cs152/sp19/lectures/L20-DSA.pdf)
- The V1 of the TPU was used for 4+ years
- [A Domain-Specific Architecture for Deep Neural Networks](https://cacm.acm.org/magazines/2018/9/230571-a-domain-specific-architecture-for-deep-neural-networks/fulltext)
- Use `Bfloat16`
- [Image Classification at Supercomputer Scale](https://arxiv.org/pdf/1811.06992.pdf)

## [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
*todo*
Mostly hardware stuff where I don't know that much atm.

## [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
*todo*
