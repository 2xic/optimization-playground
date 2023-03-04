# What accelerators are out there ? 
Knowing what accelerators are out there, and how they can be used, and what they can do will be more and more important. 

## List of papers
[AI Chip Paper List (technical papers)](https://github.com/BirenResearch/AIChip_Paper_List)

[AI Chip List (less technical)](https://github.com/basicmi/AI-Chip)

[Twitter list with a lot of papers](https://twitter.com/ogawa_tter/status/1398315188206465024)

## Talks to look at (?)
[High-Performance Hardware for Machine Learning](https://media.nips.cc/Conferences/2015/tutorialslides/Dally-NIPS-Tutorial-2015.pdf)

[Apple Neural Engine Internal: From ML Algorithm to HW Registers](https://www.blackhat.com/asia-21/briefings/schedule/#apple-neural-engine-internal-from-ml-algorithm-to-hw-registers-22039)

[Exploration and Tradeoffs of Different Kernels in FPGA Deep Learning Applications](http://www.ispd.cc/slides/2018/s2_3.pdf)

## Articles to read
[Understanding Roofline Charts](https://www.telesens.co/2018/07/26/understanding-roofline-charts/)

[An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)

--------------

## The Apple Neural Engine
Very little public information about it, and no public api afaik.
Seriously, searching for "site:machinelearning.apple.com apple neural engine" only gives information about 

[Gehot has reversed part of it](https://github.com/geohot/tinygrad/tree/master/accel/ane)

### [Apple Neural Engine Internal: From ML Algorithm to HW Registers](https://www.blackhat.com/asia-21/briefings/schedule/#apple-neural-engine-internal-from-ml-algorithm-to-hw-registers-22039)
Cool, so FaceId actually uses "Secure Neural Engine" and is also documented at [apples page on secure enclave](https://support.apple.com/lv-lv/guide/security/sec59b0b31ff/web).

They mention a tool, I have not heard of "Espresso" (*what is this ?* ).
So I found this page [A peek inside Core ML](https://machinethink.net/blog/peek-inside-coreml/) which concludes that Espresso is just a nickname for the part in [CoreML](https://developer.apple.com/documentation/coreml) that runs neural networks.

The general pipeline is described as `CoreML -> Espresso -> Apple Neural Engine Compiler (APE)`, but I have not been able to find any information about the APE. It's however discussed some in the talk (currently only looked at the slides).

Cool, they have released the tools they used [https://github.com/antgroup-arclab/ANETools](https://github.com/antgroup-arclab/ANETools) 

## Google TPUs
Supposedly based on this [lecture from Taylor University](https://www.youtube.com/watch?v=9Mo80a4s0Bs), Google started to rent out the TPUs because they had this extra compute that was not utilized always.

### [OpenTPU](https://github.com/UCSBarchlab/OpenTPU)
Theses have tried to reimplement the Google TPUs based on  [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)

### [In-Datacenter Performance Analysis (Presentation)](https://www.cse.wustl.edu/~roger/566S.s21/In-Datacenter%20Performance%20Analysis%20of%20a%20Tensor%20Processing%20Unit.pdf)
Looking at this before the paper, and quite infesting. They wanted to make interference 10x that of GPU in about 15 months. It's also interesting that 3 types of NNs represent 95 % of Google interference workload (this has probably changed now).


### [In-Datacenter Performance Analysis of a Tensor Processing Unit (Paper)](https://arxiv.org/abs/1704.04760)
*TODO*

## Tesla Dojo
- [Dojo](https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/)
- [D1 chip](https://www.datacenterdynamics.com/en/news/tesla-details-dojo-supercomputer-reveals-dojo-d1-chip-and-training-tile-module/)

*TODO*

## TPUs in general
### [Understanding Matrix Multiplication on a Weight-Stationary Systolic Architecture](https://www.telesens.co/2018/07/30/systolic-architectures/)
Most TPUs are built on top of a MXU (multiply-accumulate systolic array matrix unit).
Basically it allows more operations to happen in parallel (for instance multiply and accumulate).

So what is a systolic architectures ? It's basically a system of interconnected processing elements that will have information flown between them, and all cells are capable of some simple operation. I/O only happens at the boundary of the cells. 

*TODO: The article has some nice illustration that you should try to compress to words*

### [What’s inside a TPU?](https://medium.com/@antonpaquin/whats-inside-a-tpu-c013eb51973e)
- CPUs are scalar machine
- GPUs are vectors machine
- TPUs are matrices machine

So, how does the TPU become a matrices machine ? It's largely thanks to the systolic array architecture. Which allows computation to happen in efficiently and in parallel.

On the TPU (at least Google TPU v1) 16-bits is used to represent the number, using standard 16-bit would be problematic because of the range so Google solved this with bfloat16 (8 bits exponent instead of 5, same as a float32 but ofc less precise)

### [lecture from Taylor University](https://www.youtube.com/watch?v=9Mo80a4s0Bs)
Just looked at the [transcript](https://youtubetranscript.com/?v=9Mo80a4s0Bs), but should probably look at the video.

It's a bit hard to see the whiteboard, might revisit.

