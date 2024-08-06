## Google TPUs
Supposedly based on this [lecture from Taylor University](https://www.youtube.com/watch?v=9Mo80a4s0Bs), Google started to rent out the TPUs because they had this extra compute that was not utilized always.

### [OpenTPU](https://github.com/UCSBarchlab/OpenTPU)
Theses have tried to reimplement the Google TPUs based on  [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)

### [In-Datacenter Performance Analysis (Presentation)](https://www.cse.wustl.edu/~roger/566S.s21/In-Datacenter%20Performance%20Analysis%20of%20a%20Tensor%20Processing%20Unit.pdf)
Looking at this before the paper, and quite infesting. They wanted to make interference 10x that of GPU in about 15 months. It's also interesting that 3 types of NNs represent 95 % of Google interference workload (this has probably changed now).

### [In-Datacenter Performance Analysis of a Tensor Processing Unit (Paper)](https://arxiv.org/abs/1704.04760)
*TODO*

### [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
Using BFloat16 (Brain Floating Point Format 16) is a one of the the Google TPUs some extra high performance. Models generally need lower precisions (especially for inference only) and BFloat16 is a way to achieve "as good" results as float32, but with less bits.

[Origin story](https://twitter.com/JeffDean/status/1717030662144921670) - Jeff wanted to send fewer bits over the network to optimize the distributed training system at Google.

### [Domain-Specific Architectures for Deep Neural Networks](https://inst.eecs.berkeley.edu//~cs152/sp19/lectures/L20-DSA.pdf)
- The V1 of the TPU was used for 4+ years
- [A Domain-Specific Architecture for Deep Neural Networks](https://cacm.acm.org/magazines/2018/9/230571-a-domain-specific-architecture-for-deep-neural-networks/fulltext)
- Use `Bfloat16`
- [Image Classification at Supercomputer Scale](https://arxiv.org/pdf/1811.06992.pdf)
- https://thechipletter.substack.com/p/googles-first-tpu-architecture?utm_source=substack&utm_medium=email&utm_content=share
- 

## [Ten Lessons From Three Generations Shaped Google’s TPUv4i](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
*todo*
Mostly hardware stuff where I don't know that much atm.

## [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
*todo*

## [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
- Google has used ASICs for NN since 2006
- Uses CISC over RISC
- 

## [Google TPU v5p](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer)
- [Jeff Dean Thread](https://twitter.com/JeffDean/status/1732503666333294846)

