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

### [The Use Case for Relative Position Embeddings](https://ofir.io/The-Use-Case-for-Relative-Position-Embeddings/?utm_source=pocket_saves)
TLDR: The author want us to move away from absolute to relative position embeddings

The idea in general makes sense to me, having a fixed relative scope is nicer also. The model just need to focus on one spot which improved performance.

Referenced papers
[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/pdf/2108.12409.pdf)
[The Curious Case of Absolute Position Embeddings](https://arxiv.org/pdf/2210.12574.pdf)


## [Transformers w/ Andrej Karpathy](https://www.youtube.com/watch?app=desktop&v=XfpMkf4rD6E)
- Decade ago: 
  - Feature detectors used everywhere
  - separates models depending on context (nlp / cv etc)
- Now
  - Transformer - single architecture everywhere
- Timeline to the transformer
  - 2003
    - Neural probabilistic language model
  - 2014
    - Sequence to sequence learning with Neural Networks
    - Neural machine translation by jointly learning to algin and translate
      - [Dzmitry Bahdanau](https://rizar.github.io/)
  - 2017
    -  Attention is all you need
    -  


### [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/pdf/2310.06816.pdf)
- Assumes black-box access to the model
- Tested with text that are 32-tokens in length
- Iterative process where the model is able to find embeddings that correlate to the original text

### [Grok](https://x.ai/)
- LLM that has a better sense of humor than most other LLMs (not that I'm been able to test it yet)
- It also has "real-time" knowledge of what is happening based on data from the X platform (is the model just using some apis internally ? )
- The model looks some what promising on the dataset

What is more interesting is the engineering section
- They use JAX, Rust and Kubernetes for the training and inference stack. They also expand on that in the [careers](https://x.ai/career/) page
- They also use triton
  

### [Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models](https://arxiv.org/pdf/2311.00871.pdf)
- They find limited evidence that LLMs are able to generalized being the training dataset (this isn't really any news?)
- They do however do well on tasks defined within the training data
- 

### [Gecko: Versatile Text Embeddings Distilled from Large Language Models](https://arxiv.org/pdf/2403.20327)
- Google model.
- They use a LLM to create the embeddings from by distilling it. 

## OpenAI api changes
- [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Realtime API](https://openai.com/index/introducing-the-realtime-api/) improves the speech to text > completion > text to speech flow
- [Model distillation in the API](https://openai.com/index/api-model-distillation/) improved api infra for distilling openai models.

### [Zamba2-7B](https://www.zyphra.com/post/zamba2-7b)
- SOTA small model
- [Improvement over the old version](https://arxiv.org/abs/2405.16712)
- [Mamba2 Blocks](https://arxiv.org/pdf/2405.21060)

## [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/pdf/2408.03314)
- The idea behind strawberry, instead of spending time training, spend more time thinking before outputting answer.
- [Searching](https://arxiv.org/pdf/2408.03314#page=8) for the best answer.
- How do you select the time to spend here vs training time ? 

## [Badllama 3: removing safety finetuning from Llama 3 in minutes](https://arxiv.org/pdf/2407.01376)
It's easy to remove safety features when you have access to the model.

## [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/pdf/2504.13837)
[Website](https://limit-of-rlvr.github.io/)

The author conclusion suggests that RL doesn't seem to cause LLM to acquire better reasoning capabilities, but instead it makes them more sample efficient. Distilled models on the other hand, seems to improve the reasoning capabilities of the base model.


