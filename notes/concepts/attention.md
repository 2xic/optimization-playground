# Attention

There are many flavours of attention and new onces will be created in the future.

## Flash attention
[Paper](https://arxiv.org/pdf/2205.14135)

[ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

Compared to the original attention this version is
- Fast (about 15% faster than SOTA at the time)
- Memory-efficient by being linear instead of quadratic on time.
- Exact - it's not a approximation method, it's the same as the original version

What is the trick ? Being IO aware. Attention is memory bound. In addition, faster memory usually have less capacity. Goal is therefore to reduce communication of memory.
The trick is using kernel fusion to optimize this communication. 

[Triton implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py).

[Philippe Tillet](https://www.mlsys.ai/papers/tillet.html) also tried to discover something similar.

## [FlashAttention-2](https://crfm.stanford.edu/2023/07/17/flash2.html)
Optimize the work partitioning.

