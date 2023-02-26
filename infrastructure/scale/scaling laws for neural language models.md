### [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)

- Scaling laws for language models performance with cross-entropy loss
  - The loss scales as a power-law with model size, dataset size, and the amount of compute used for training 
- Focus on the transformer architecture

**Key findings**
- Performance depends strongly on scale, and weakly on model shape
  - Scale: Number of model parameters(excluding embeddings), size of dataset, and amount of compute
    - Known as the scaling factors
  - Shape: Depth vs width 
- Power laws: Performance has a power-law relationship with the scaling factors
- Performance improved predictably if model parameters and size of dataset is scaled up
  - BUT gives diminishing return if one of them is held back, and hte other is increased
- Larger models are more sample efficient

**Setup**
- Model trained on WebText2 with Vocab size = 50257
- Byte-pair encoding
- Adam optimizer

----

### [https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt)
- Generic foundation models can be created by having large language model train on a large training corpus, and what is also interesting is just how much better GPT-3 is at various downstream task over the previous GPT-2.
- Rest of the blog mostly talks about the scaling paper, but that is mostly cover in the section above here.
