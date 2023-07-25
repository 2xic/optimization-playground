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


### [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426.pdf)
Deep learning is great, but it's computational expensive. The authors show that gzip + k-nearest neighbor gives good results fro text classifications.

One way of doing this kind of compression is `Compressor-Based Text Classification` where one check the probability distribution between a class of documents and the given document

The authors does the following
1. Load `X, y` dataset and iterate over the entire dataset (do it in two loops)
2. Compress input sentence `X_1`
3. Compress input sentence `X_2`
4. Compress combined sentence `X_1 + X_2`
5. Then they compute the "Normalized Compression Distance"
   - $$\frac{(C(X_1 + X_2) - min(C(X_1), C(X_2)))}{max(C(X_1), C(X_2))}$$
6. Sort this score index with the associated label which 
7. Load `n` of these and select the top score

The model itself does quite okay and on par with several DL models, but most of those evaluated are relatively old.

The code itself is just 14 lines which is the most inresting part of this. 


### [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://icml.cc/media/icml-2022/Slides/17378.pdf)
Training a multiple models to combat costs. Creates a simple layer that routes the input to a selected numbers of experts. The final prediction is the weighted results of these experts.
