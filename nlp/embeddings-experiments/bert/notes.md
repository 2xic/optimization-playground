
## [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- The input is kinda interesting, they combine the embedding, position adn segmentation
- 

### Difference between bert and GPT
Bert uses encoder instead of decoder Which could results in token leakage [1], but BERT get's around this with random masking (ref [1]) 

[1] https://www.kaggle.com/code/residentmario/notes-on-gpt-2-and-bert-models


## masking
The core idea of BERT is related to masking the words and have the model learn to understand what words are missing.

[Masking](https://neptune.ai/blog/unmasking-bert-transformer-model-performance)
- Using `K=15%` for masking words, more looses context and less to easy.
  - https://arxiv.org/pdf/2202.08005.pdf
    - Notes on if `15%` is a good limit, might not be.
- https://nn.labml.ai/transformers/mlm/index.html
  - 

### Implementations
https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model
