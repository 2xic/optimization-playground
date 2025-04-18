Some good links 
- [Development speed over everything - blog.comma.ai](../iteration-speed/Development%20speed%20over%20everything.md)

# [Thoughts on Model Training Infrastructure](https://xander.ai/thoughts-on-ai-training-infrastructure)
- Depending on your computer scale, you might need more tricks to get the best results
  - [DiLoCo](https://arxiv.org/pdf/2311.08105.pdf) might be one trick Google uses for Gemini

## Cluster management
- Be prepared for hardware failures
  - [Jeff dean](https://perspectives.mvdirona.com/2008/06/jeff-dean-on-google-infrastructure/) also mentioned this in some talk. Hardware failure in isolation is uncommon, but the more computers you have the more common it becomes
  - You don't want your entire cluster to fail if one computer fails

### Monitoring
- DataDog 
- `nvidia-smi` is bad, use something like tensor core utilization instead.
- Use good utilization metrics, i.e MFU from [PaLM](https://arxiv.org/pdf/2204.02311.pdf) paper.

### Experiment Layer
- Job scheduling should be abstracted away and same is true for metrics collections.
- Automated loss spike recovery, when something goes long the model should recover.
- Think about what should be considered show stopping errors
  - "Hardware is too expensive to not make use of it when non-critical components are down."
- Performance visualization
- **Iteration rate over anything**

# [Scaling Kubernetes to 7,500 nodes](https://openai.com/research/scaling-kubernetes-to-7500-nodes)
[Scaling Kubernetes to 2,500 nodes](https://openai.com/research/scaling-kubernetes-to-2500-nodes)
- "largest-scale workloads manage bare cloud VMs directly"
- Kubernetes clusters are nice for quick experimentation.

# [Distributed deep learning](https://id2223kth.github.io/slides/2022/12_distributed_dl.pdf)
- Data Parallelization 
  - Run model on each device and send the data to those models and average over that
  - Challenges
    - Aggregation and the synchronization mechanisms
- Model Parallelization
  - Model is split over multiple devices. Which can be tricky depending on the aritechture.

There is also a good blogpost by [Lilian Weng](https://lilianweng.github.io/posts/2021-09-25-train-large/) on this topic and they also posted it on the [OpenAi blog](https://openai.com/research/techniques-for-training-large-neural-networks).

## [Scaling AI Infrastructure at OpenAI](https://www.youtube.com/watch?v=cK7qFZ9J6k0)
- Talks about the Kubernetes setup

## [Reinforcement Learning on Hundreds of Thousands of Cores](https://www.youtube.com/watch?v=ui4F_A46wN0)
- OpenAI and the Redis setup they used for Dota 2

## DeepSeek
- [Fire-Flyer File System (3FS)](https://arxiv.org/pdf/2408.14158) which they have [also open sourced](https://github.com/deepseek-ai/3FS) with associated [design notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md).
  - Parallel file system architecture which allows multiple algorithms to read the data without contentions.
  - Uses RDMA and SSds for storage. Uses foundationdb as it's distributed storage layer.
  - [Blogpost on Fire-Flyer](https://www-high--flyer-cn.translate.goog/blog/3fs/?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp) 
- [An Intro to DeepSeek's Distributed File System](https://maknee.github.io/blog/2025/3FS-Performance-Journal-1/)
