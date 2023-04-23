## Natural language processing
For the paper notes that are so short that they don't need their own file.

### [Evaluating Word Embedding Models: Methods and Experimental Results](https://arxiv.org/pdf/1901.09785.pdf)
First it gives a nice introduction to some existing methods to create embeddings.

All methods gives out different embeddings vectors, but there is some properties they all should share
- Non-conflation
  - Small context differences should results in a different vector
- Robustness Against Lexical Ambiguity
  - "the **bow** of a ship" and "**bow** and arrows" have different meaning
- Reliability
    - Even if the model is retrained the output should be stable
- Good Geometry
  - Distance should be good

They mention and show the method `3CosMul` for finding analogies, I have not heard about this before. Maybe worth looking into.

### [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
The technique is as follows
1. Pretrain the model on a general dataset (Wikitext for instance)
2. Target task fine tuning by using two methods
  - Discriminative fine-tuning which uses a different learning rate for all layers.
  - Slanted triangular learning rates which first increases the learning rate a lot (linearly), and then have a linear decay
3. Target task classifier tine-tuning
   - Adds two new linear layers 
   - Gradual unfreezing which means that we don't finetune all layers as one, but instead gradually unfreze each layer

The analysis shows that the methods work well in practice.

