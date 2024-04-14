# TPUs in general
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

### [The Design Process for Google’s Training Chips: TPUv2 and TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
*TODO*

### [TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings](https://arxiv.org/ftp/arxiv/papers/2304/2304.01433.pdf)
*TODO*

# XLA compiler
From the official documentation [here](https://cloud.google.com/tpu/docs/intro-to-tpu#xla_compiler) they say the ML graph has to be compiled with XLA to be possible to run on the TPU.

There is some docs on XLA [here](https://www.tensorflow.org/xla). ACtually it is even better described [here](https://www.tensorflow.org/xla/architecture). The input to XLA is what is refereed to as "HLO IR" (high level optimizer intermediate representation). 

What does XLA do ? Optimization like common subexpression elimination which eliminates identical expression with a single variable. It also uses target-specific code operation by using LLVM for low level IR optimization for the generated code.

[MLIR](https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf) they go from a tensor flow graph into a xla hlo into llvm into machine ir into asm.
[MLIR Tutorial](https://llvm.org/devmtg/2019-04/slides/Tutorial-AminiVasilacheZinenko-MLIR.pdf), traditional model is AST -> LLVM. MLIR is about operations not instructions and they are kinda complex just looking at one.
[Building domain-specific compilers quickly with MLIR compiler infrastructure | Chris Lattner](https://www.youtube.com/watch?v=5OSP5DNAozU) it has little todo with ML btw. 


### [https://llvm.org/devmtg/2019-04/slides/TechTalk-Joerg-Automated_GPU_Kernel_Fusion_with_XLA.pdf](Automated GPU Kernel Fusion with XLA)
The slides by themselves were not that useful.This lead me however to find [this](https://arxiv.org/pdf/2301.13062.pdf) paper which is actually quite nice.

### [Learning to Fuse](http://mlforsystems.org/assets/papers/neurips2019/learning_abdolrashidi_2019.pdf)
Use ML to learn when to fuse - kinda cool.

# Other recourses
- [Geohot again did some (small) documentation on this](https://github.com/tinygrad/tinygrad/tree/a8f2c16f8e1670ce199b068a771b9b0d6f7ba7df/extra/accel/tpu)
- [OpenXLA](https://github.com/openxla/xla) which is an open source compiler
- [SysML 18: Jeff Dean, Systems and Machine Learning Symbiosis](https://www.youtube.com/watch?v=Nj6uxDki6-0)
