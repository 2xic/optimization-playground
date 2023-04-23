Testing various embeddings for documents.


-> Here are some models and methods listed https://lilianweng.github.io/posts/2017-10-15-word-embedding/
-> Docs embedding https://radix.ai/blog/2021/3/a-guide-to-building-document-embeddings-part-1/

### SkipGram
You predict context words based on a input word.

The model should learn to output context window probabilities.

```
python3 train_skip_gram.py
```

### CBOW
Context is input to model, and output is the main words.

You use the sum of the contexts that are then fed into an embedding layer which outputs the probabilities.

```
python3 train_cbow.py
```
