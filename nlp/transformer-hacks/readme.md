Meant to be transformer version of [gan-hacks](/gan/gan-hacks)

## Good resources
- [Converting a From-Scratch GPT Architecture to Llama 2](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb?trk=public_post_comment-text)
- [LLAMA source code](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

## Embeddings
- [LLM2Vec](https://arxiv.org/pdf/2404.05961)
  - Remove the mask from the attention training
  - Use [MASK] tokens
  - Contrastive learning
- 

## Post training
- [Speculative sampling](https://jaykmody.com/blog/speculative-sampling/)

## Interference
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

## TODO
- Benchmark [DeepSeek v3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) layers also
- Implement more transformer layer variants
  - [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
  - [Build a Large Language Model ](https://github.com/rasbt/LLMs-from-scratch) has a lot of good resources also.
- [Sliding window attention implementation](https://amaarora.github.io/posts/2024-07-04%20SWA.html#sliding-window-attention-in-pytorch)

## Generation
[KV cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
- Just a trick to optimize the generation speed.
- [ Transformers Optimization: Part 1 - KV Cache ](https://r4j4n.github.io/blogs/posts/kv/)
