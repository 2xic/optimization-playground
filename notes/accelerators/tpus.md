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
