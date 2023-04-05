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

[Large language models are having their Stable Diffusion moment](https://simonwillison.net/2023/Mar/11/llama/) thanks to [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) allowing llama to run on a cpu. It's a plain C++ implementation, and uses 4-bit quantization to reduce the memory load and allows it to run on less powerful hardware.

### [Theory of Mind May Have Spontaneously Emerged in Large Language Models](https://arxiv.org/pdf/2302.02083.pdf)
- This one is interesting, humans have the ability to somewhat understand other people mental state, or at least understand that people view the world differently. 
- Previous models (before 2022) have not have this ability, and done poorly on evaluation tasks for it. However after 2022 models released by OpenAI have started to do well on this kind of tasks.

### [Large Language Models Can Self-Improve](https://arxiv.org/pdf/2210.11610.pdf)
- Shows that LLMs can improve with unlabeled data (by using self supervised methods).
- Uses techniques like majority voting to learn (self consistency)

### [LLMs: The Important Ingredients](https://xander.ai/llms-the-important-ingredients)
What do you need to "bake" a LLM ? 
- Scale, but not just stacking GPUs, you need to have a efficient and optimized interference and training loop.
- Human preference alignment (like OpenAi did with RL)
- You need to have a large context window
- Flywheel -> deploy and improve based on feedback (including jailbreaks).

They also references many papers that is worth looking more deeply into.

-  [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf) (notes below)
- [Chain of Hindsight Aligns Language Models with Feedback](https://arxiv.org/pdf/2302.02676.pdf) (notes below)
- [Universal Transformers](https://arxiv.org/abs/1807.03819) (todo)


### [Improving language models by retrieving from trillions of tokens](https://arxiv.org/pdf/2112.04426.pdf)
- Uses 25x less parameters than models like GPT-3, but is still able to do results like GPT-3.
- Sounds like the technique they use is loading tokens based on local similarly to preceding documents. 
    - My automatic reaction to this is that the model would be more likely to overfit in the domain, but maybe that is not the case.
- The architecture seems a bit complicated thought
    - They use a frozen bert model to create embeddings
    - Text tokens are fed into this model and the document embeddings database to find similar text. 
    - This is then again fed into a transformer encoder, and part of the text chunks will then be replaced with the new embeddings with short distance (if I understand correctly)
- Hm, I take back what I said above, it's an interesting idea.


### [Chain of Hindsight Aligns Language Models with Feedback](https://arxiv.org/pdf/2302.02676.pdf)
- Models like ChatGPT uses variants of supervised learning to learn from annotated feedback (I know it's RL, but it's needs direct annotated feedback )
- The core idea of the paper is to just say "this is good/better" and "this is bad/worse" depending on the model output.
    - The idea is basically that the model based on all of this feedback should be able to learn what is considered good / bad
- Hm, I mean they still use labels, but the labels seems like they might be more relative than absolute. 

### [Program-aided Language Models](https://arxiv.org/pdf/2211.10435.pdf)
Makes the LLM use a tool to offload computation as the models are usually better at reasoning than they are at doing things like math.

Basic idea is just to tell the model to use python variables, and run the result in python (for instance).

### [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/pdf/2112.09332.pdf)
- Give GPT-3 access to a web browser to answer long formed questions
    - Basically create a simple syntax for different commands the model can "execute" and watch for them in the model output
- Then evaluate the answers based on human feedback, makes sense.
- Does not do well on out of distribution questions, but does well on ELi5 dataset.


### [github.com/transmissions11/flux](https://github.com/transmissions11/flux)
Quite cool tool that allows you to explore multiple output path for a LLM.

Basic a powertool to make it easier to interact with LLMs.

[Tweet thread from transmissions11](https://twitter.com/transmissions11/status/1640775967856803840)
