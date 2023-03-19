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

### GPT-v4 
GPT-4 is out [paper](https://cdn.openai.com/papers/gpt-4.pdf) has to be read.
So the first thing is that the new model is now a MLLM model, so it's able to take both images and text as input, and gives text as output (following the trend set by other models).
Core part of the project was the infrastructure and optimization methods to give predictability, and they were able to infer (in some aspects) how the model would behave on as little as 1 / 1000th of the compute of the final model.
[Benchmarks](https://twitter.com/SilasAlberti/status/1635693275230380032) show that it outperforms existing LLMs, and it also does well on exams (top 10% vs bottom 10% on GPT 3.5).
GPT-4 still suffer from hallucinations, has a limited (but increased [4096 to 32k](https://twitter.com/nbashaw/status/1635689092515233792)) context window and still does not learn from experience.
The training steps seems similar to that of ChatGPT by having Reinforcement Learning from Human Feedback. They share no other details on the model because of "safety and competition"
Predictable scaling by using scaling laws to predict how the model would act.
**They also tested GPT-4 on [Inverse scaling prize (tasks that should be harder for LLMs, but easy for small models)](https://github.com/inverse-scaling/prize), and it odes amazingly well (looks like 100% accuracy)**

- [Seems overfited on Codeforces](https://twitter.com/cHHillee/status/1635790330854526981) quote from the thread `So... the fact that it solves 10/10 problems from pre-2021 and 0/10 of the most recent problems (which it has never seen before) is very suspicious.`
- [Can GPT-4 *Actually* Write Code?](https://tylerglaiel.substack.com/p/can-gpt-4-actually-write-code) concludes that GPT-4 cannot "solve" problems outside the training dataset. 


