## [Distributed deep learning](https://id2223kth.github.io/slides/2022/12_distributed_dl.pdf)
- Data Parallelization 
  - Run model on each device and send the data to those models and average over that
  - Challenges
    - Aggregation and the synchronization mechanisms
- Model Parallelization
  - Model is split over multiple devices. Which can be tricky depending on the aritechture.

There is also a good blogpost by [Lilian Weng](https://lilianweng.github.io/posts/2021-09-25-train-large/) on this topic and they also posted it on the [OpenAi blog](https://openai.com/research/techniques-for-training-large-neural-networks).
