I'm trying to replace the OpenAI embeddings api with something custom.

```bash
python3 -m optimization_playground_shared.embedding_test.model_1
python3 -m optimization_playground_shared.embedding_test.model_2
python3 -m optimization_playground_shared.embedding_test.model_3
python3 -m optimization_playground_shared.embedding_test.model_4
# 
python3 -m optimization_playground_shared.embedding_test.hosted_trained_models
```

## Papers with embeddings
- https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf
- https://lilianweng.github.io/posts/2017-10-15-word-embedding/
- https://datascience.stackexchange.com/a/128401
- https://paperswithcode.com/task/document-embedding

### Matryoshka Representation Learning
- https://arxiv.org/pdf/2205.13147
- https://huggingface.co/blog/matryoshka

### Latent representation
- https://arxiv.org/pdf/2006.07733
  
## Ideas
Most deep learning models are based on context models. Things like bert, word2vec, etc. tries to predict the missing word in a context and this way learn,. Most of these use negative sampling loss. 

Some ideas
- using tf-idf as anchor in the loss
- sample a document into paragraphs and ask the model if the two sequences are part of it or not.
- train gpt like model and just the things mentioned above during post-training
- train gpt like model and fine-tune it using some ranking algorithms (RankNet, etc).

## Training larger models
- Mixed Precision
  - https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
  - https://github.com/NVIDIA/apex
- https://pytorch.org/docs/stable/distributed.elastic.html
- Profiling
  - https://huggingface.co/blog/train_memory?utm_source=pocket_shared
  - 

## Pre-training 
Using [applied/embeddings-documents-viz](applied/embeddings-documents-viz) to visualize the model output (the raw next token prediction model) without any fine tuning. 

You can clearly see that our model is packed together a lot more than the other reference models.

![vis](./readme/embeddings_viz.png)

Doing some post training on the model results in the following:

![vis](./readme/embeddings_viz_post_training.png)


## Evaluations
- https://zilliz.com/learn/evaluating-your-embedding-model
- https://www.cambridge.org/core/services/aop-cambridge-core/content/view/EDF43F837150B94E71DBB36B28B85E79/S204877031900012Xa.pdf/div-class-title-evaluating-word-embedding-models-methods-and-experimental-results-div.pdf
