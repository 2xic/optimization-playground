
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

## wordpiece
This is another idea that Bert and a lot of other models are now using where they split up the tokens.
It's introduced in [this](https://arxiv.org/pdf/1609.08144v2.pdf) paper from Google and taking the papers example
```
Word : Jet makers feud over seat width with big orders at stake
Wordpiece : : _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake
```
Basically words are split into tokens and we have `_` as the special token indicating start of a word. 

It's framed as an optimization problem where the model is allowed set to use `D` tokens. The optimization problem will then try to select the best word split for highest accuracy.

They also references these papers for more details
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf) which has code example
- [Japanese and Korean voice search](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)

### Implementations
https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model
