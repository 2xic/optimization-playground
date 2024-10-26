## [Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss](https://arxiv.org/abs/2410.17243)
- The vanilla implementation of contrastive loss is `O(b ** 2)` where `b` is the batch size.
- Proposed method is `O(b / n ** 2)` where `n` is the number of GPUs.
- Leverages the log-sum-exp cumulative property to break the computation into multiple blocks that can be done on multiple GPUs at the same time. The method is described [here](https://arxiv.org/pdf/2410.17243#page=4).
-  
