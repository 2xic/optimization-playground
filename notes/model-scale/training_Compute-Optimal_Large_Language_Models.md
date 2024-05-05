## [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
- Investigates how to scale a (transformer) model for a given compute budget. They believe that most models are undertrained because models have been scaled without also scaling up the data
- They believe that if you double the model parameters, you should also double the amount of tokens you train on.
- To verify the hypothesis they train a model based on the prediction they have. The results is that by having less parameters than other models, but more data they are able to outperform other famous models like GPT-3 on various evaluation tasks.

