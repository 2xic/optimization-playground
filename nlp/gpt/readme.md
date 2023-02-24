## GPT-v1

[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
[Blog](https://openai.com/blog/language-unsupervised/)

- Learning from unlabeled text is challenging
  - Unclear what the optimization objective that is most efficient for learning text representation that can be transferred
  - No consensus on the most effective way to transfer learned representation to a target task
- Approach in the paper 
  - First learn a representation with unlabeled data. 
  - Then finetune on the task
- Model
    - h_0 = UW_e + W_p
      - U is the context vectors
      - W_e is token embedding matrix
      - W_p is position embedding matrix
    - h_x = transformer_block
      - From Multi-layer Transformer decoder
        - https://arxiv.org/pdf/1801.10198.pdf
          - Section 4.2.4
          - Fig. 1
        - https://ulaval-damas.github.io/glo4030/assets/slides/3.1-Transformer-Decoder.pdf
    - Softmax(h_nW_e^T)
- Supervised learning
    - You finetune a new layer w_g
    - See fig 1.
- 


## GPT-v2 (todo)
TODO: Look at this blogpost
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
[Blog](https://openai.com/blog/better-language-models/

TODO: Look at this blogpost
[https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)

[https://jaykmody.com/blog/gpt-from-scratch/](https://jaykmody.com/blog/gpt-from-scratch/)
- Builds up the concept of the GPT architecture from a simple input and output model, from vocab encoding, and various activations.
- Includes the positional embedding logic, but not the same attention layer as in the actual gpt model, but the classical attention is all you need method
- Still pretty nice blogpost and nice code [https://github.com/jaymody/picoGPT/blob/main/gpt2.py](https://github.com/jaymody/picoGPT/blob/main/gpt2.py)

[https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- Builds up an intuition about how the chatgpt model works from a n-char probability model to a n-word probability model.
- In addition he has some nice visualizations for the attention layers
- 

