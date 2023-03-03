## Large language models

### [Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761.pdf)
Language models are good at a lot of complicated tasks, but somtimes not the very good at things like simple math questions. The authors therefore learned the model to use a API to ge tthe best of both worlds.
The way this is done is give the LLM a few exampels of how to use the tool, and have an external script be able to handle the results.

### [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
Show that models are able to reason by just prompting it with "let's think step by step".

### [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)
Shows that using automated generated prompts outperforms the prior baseline.
The automatic prompt engineer is trained by given an demostration (for instance some input + output), and the LLM outputs a set of instructions candiates, the output is scored and filtered then executed and evaluated.

## [Modern LLMs: MT-NLG, Chinchilla, Gopher and More](https://cameronrwolfe.substack.com/p/modern-llms-mt-nlg-chinchilla-gopher)
- Talks about various other large language models compared to GPT-3.
- Megatron is a 530B parameters model (GPT-3 has "only" 175B), and does well on various tasks. Except on a reading comprehension task with few show ( Race-h ), interestingly enough.
    - It' also worth noting that a supervised model still beats Megatron on specific tasks, but one can clearly see (even if marginal) that bigger models produces better results.
- It's also interesting that even larger llms have problems with things like math (I guess it's not that weird given the loss function they are optimizing for, but still the fact that models with fewer parameters can win over one with more parameters in categories like math is a bit strange).
- The article also talks about the infrastructure/setup needed to train so large language models. Mainly/Namely distributed compute and many gpus. Many of the papers also experiences some of the effects noticed for scaling transformer models (i.e that data + models have to be scaled for best results). 

### Offensive and Defensive Uses of Code LLMs (talk by Xander Dunn)
[video](https://drive.google.com/file/d/1KpBjboIDy-pFbzj3s1UpnZKJCVyfz0x7/view)
[slides](https://docs.google.com/presentation/d/1XSxPfZpo_9h2G6ICYYf6BuMhBjQbAAtuAw_hRz94huY/edit)
Interesting talk, especially since the models are not trained (specifically) for this task, but still does it quite well.

There are two papers he also brought up that i found interesting
[Pop Quiz! Can a Large Language Model Help With Reverse Engineering?](https://arxiv.org/pdf/2202.01142.pdf)
- Gives the OpenAi codex model various code questions for various C code
- For instance one of the code samples have a "evil" program, and the authors quiz Codex about it (pop-quiz). It does quite good, but the authors have to play a bit with the temperature parameter.
- While the model does not do so well on the quiz when the output is decompiled and stripped of symbols it still is able to reason some about the code. 
- It's also able to reason about software like PID controllers, cool.

[Examining Zero-Shot Vulnerability Repair with Large Language Models](https://arxiv.org/pdf/2112.02125.pdf)
- Tries to answer the question: can LLMs be used to fix up security problems ? 
- Surprisingly (?) LLms are also capable of doing this quite well.

### [LLaMA](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)

[@guillaumelample thread](https://twitter.com/guillaumelample/status/1629151231800115202)
- Beats models like GPT-3 on most benchmarks
- Trained on public data, and can therefore be open source and reproducible (hopefully). 
- Trained on 1T tokens, and even with that the 7B parameters model was still improving

Looks like the claim of open source is lie [https://twitter.com/balajis/status/1629449439726415872?cxt=HHwWgMDRpbvo-5wtAAAA](https://twitter.com/balajis/status/1629449439726415872?cxt=HHwWgMDRpbvo-5wtAAAA)

Code is open [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)

[kaushikpatnaik thread](https://twitter.com/kaushikpatnaik/status/1629194428312330240)
- Conclusion is that LLMs are still undertrained.
- References [chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) which was an interesting post on how data is the bottleneck and not the size of the model.

### [Theory of Mind May Have Spontaneously Emerged in Large Language Models](https://arxiv.org/pdf/2302.02083.pdf)
- This one is interesting, humans have the ability to somewhat understand other people mental state, or at least understand that people view the world differently. 
- Previous models (before 2022) have not have this ability, and done poorly on evaluation tasks for it. However after 2022 models released by OpenAI have started to do well on this kind of tasks.

### [Large Language Models Can Self-Improve](https://arxiv.org/pdf/2210.11610.pdf)
- Shows that LLMs can improve with unlabeled data (by using self supervised methods).
- Uses techniques like majority voting to learn

### [LLMs: The Important Ingredients](https://xander.ai/llms-the-important-ingredients)
What do you need to "bake" a LLM ? 
- Scale, but not just stacking GPUs, you need to have a efficient and optimized interference and training loop.
- Human preference alignment (like OpenAi did with RL)
- You need to have a large context window
- Flywheel -> deploy and improve based on feedback (including jailbreaks).
- 

He also references many papers that is worth looking more deeply into.

[Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)
[Chain of Hindsight Aligns Language Models with Feedback](https://arxiv.org/pdf/2302.02676.pdf)
[Universal Transformers](https://arxiv.org/abs/1807.03819)



