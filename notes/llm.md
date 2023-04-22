## Large language models
For the paper notes that are so short that they don't need their own file.

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

[Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://vicuna.lmsys.org/) fine tuned version of LLAMA based on [ShareGPT](https://sharegpt.com/). [Code](https://github.com/lm-sys/FastChat)


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
    - The model is "full control" over the navigation and processing of the site by using these commands
- Then evaluate the answers based on human feedback, makes sense.
- Does not do well on out of distribution questions, but does well on ELi5 dataset.


### [github.com/transmissions11/flux](https://github.com/transmissions11/flux)
Quite cool tool that allows you to explore multiple output path for a LLM.

Basic a powertool to make it easier to interact with LLMs.

[Tweet thread from transmissions11](https://twitter.com/transmissions11/status/1640775967856803840)

### [Supercharger: Offline Automatic Codegen](https://catid.io/posts/auto_codegen/)
With a few hacks you are able to use the LLm to help you write and clean up code. 
- They used the `AST` module to fix syntax errors
- They run the code in docker to prevent accidental LLMS "attacks" (rm -rf has no effect on host)
- Use the LLM to score the code.

[Code](https://github.com/catid/supercharger)

### [Eight Things to Know about Large Language Models](https://cims.nyu.edu/~sbowman/eightthings.pdf)
My takeaway from what the authors 
- Most of the techniques to train new LLms are mostly the same (as can be seen by for instance LLAMA), but the amount compute applied has changed for some models.
- There is evidence that LLMs build some kind of internal representation of the world (they can follow object locations etc).
- Nobody knows what's going on inside a LLM (like most other NN)

### [Schrodinger's Riddle](https://twitter.com/dylanhendricks/status/1642939372306669568)
- GPT-4 playing 20 questions
- GPT-4 cannot play 20 questions, it just hallucinates a answer
- The comment section is interesting, people tried things like pre-commits, but ofc it does not work.

### [A Watermark for Large Language Models](https://arxiv.org/pdf/2301.10226.pdf)
- Can watermark without access to model parameters or api
- The algorithm works on as little as 25 tokens
- The detection algorithm could also be made public
- Watermarking low entropy text is hard (i.e computer code can be written by man and machine, and it's sometimes hard to tell the difference in the code in question is a for loop)

Simple proof of concept watermark algorithm
0. Initialize the prompt
1. Get the probability disturbing for the next token 
2. Compute hash of the previous token, and use it to seed the random number generator
3. Using the seed randomly color the vocabulary into green and red of equal size
4. Sample from the probability distribution based on what is in the green section

-> Meaning you only need to know one token to run the test on the vocab.
-> However, you still need to know the entire vocab though. Right ? They don't make this very clear, but we need th vocab to recreate the green / red list.

Improved algorithm
- Just apply a small $\sigma$ on the logits before the softmax in the last layer to any green tokens instead of zeroing all the red tokens to 0
  
### [Can AI-Generated Text be Reliably Detected?](https://arxiv.org/pdf/2303.11156.pdf)
- paraphrasing attacks -> small rephraser on top of the generative output 
  - Seems to break most detectors
- They also show that for a sufficiently good LLM most detectors have to results to being just a random classifier. 
  - Same problems as GANS, makes sense, the more human the LLM the harder it is to detect.
  - Just running `PEGASUS` transformer on the output makes the watermark "disappear".

### [On the Possibilities of AI-Generated Text Detection](https://arxiv.org/pdf/2304.04736.pdf)
- Unless the distribution of the LLMs and human are the same, it should be possible to detect
- References this paper [DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature](https://arxiv.org/pdf/2301.11305v1.pdf)
  - Basically looks at permutations of the model (GPT-3) generated text vs the original text
- The main conclusion they have, with more model sample you can more confidently say if the text is generated by human or machine
- 

### [Teaching Large Language Models to Self-Debug](https://arxiv.org/pdf/2304.05128.pdf)
- You can teach an LLM to do RubberDuck debugging to make it solve the problems it faces with few-shot examples
  - Simple feedback : works / does not work
  - Unit tests feedback can be used
  - Code explications can be used (the model explains the code)
  - 

### [Task-driven Autonomous Agent Utilizing GPT-4, Pinecone, and LangChain for Diverse Applications](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)
- Thread on [twitter](https://twitter.com/yoheinakajima/status/1640934493489070080?s=20)
- Uses https://www.pinecone.io/
- The graph
  - User Provides the task and objective
  - Execution agent (GPT-4) stores the task and result pair
  - Task creation agent takes the result and memory context
  - Adds the task to the task queue which is then read from by Task Prioritization Agent, 
- [code](https://github.com/yoheinakajima/babyagi)

There is also this [auto-gpt](https://stablecog.com/blog/what-is-auto-gpt) which does something similar. [Code](https://github.com/drakyanerlanggarizkiwardhana/Auto-GPT)

### [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/pdf/2303.17580.pdf)
- Reminds me of [Prismer: A Vision-Language Model with An Ensemble of Experts](https://arxiv.org/pdf/2303.02506.pdf)
- Basically LLM does planning, and submodels does execution. Results is then compressed by LLM.
  - The core logic is split into    
    - Task planning
    - Model selection
    - Task execution
    - Response generation
- The example selection is pretty cool though
- [code](https://github.com/microsoft/JARVIS)

### [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/pdf/2303.12712.pdf)
- [I saw this tweet thread mention the paper](https://twitter.com/mmitchell_ai/status/1645571158585253888)
  - Basically they are most likely not testing things outside the dataset
- Just a hype paper, not reading

### [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496.pdf)
1. Given a query, it is given to a instruction model like InstructGPT to generate a hypothetical document. 
2. The document can contains things that are wrong, but has some relevant patterns
3. Then an (unsupervised) encoder encodes the document to a embedding.
4. Good (real) document candidates are then found based on vector similarly to the embedding

### [On the Sizes of OpenAI API Models](https://blog.eleuther.ai/gpt3-model-sizes/)
Interesting, they use [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to be able to estimate the model size.
Match up the task to other models, and or gpt-3 paper and then you have an estimate.

### [Prompt injection: what’s the worst that can happen?](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/?utm_source=pocket_saves)
There is also this video from [liveoverflow](https://www.youtube.com/watch?v=Sv5OLj2nVAQ)

Summary
- Prompt injection is a thing, and most devs (probably) don't think enough about this yet.
- It is also a security problem that is hard to create a [soloution](https://simonwillison.net/2022/Sep/16/prompt-injection-solutions/) for, since the ML models are black boxes. 
  - Probably with enough fine-tuning you could abolish a large part of the problems

### [Is ChatGPT a Good Recommender? A Preliminary Study](https://arxiv.org/pdf/2304.10149.pdf)
Quickly glanced over the paper

Input is usually a context list of what the user previously have interacted with and liked as context. Then ChatGPT is told what are candidates for recommendations

-> ChatGPT has a problem ranking recommendations from a list (i.e sorting them based on score)
-> ChatGPT can give a direct recommendation from a list (give output recommendation, but not sort it)

### [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)
Quickly glanced over the paper

NLP has many applications in Finance (sentiment, Q & A, summary, etc). How does a GPT like model act trained on a 363 billion token finance dataset ? 
The model is trained by Bloomberg which is cool. 

The results are that the model does very well on NLP in finance.
- Does well on news sentiment datasests (best model), but it has not been compared to GPT-3 
- 
- It knows the ceo of companies better than models like GPT-NeoX and FLAN-T5-XXL
- Does well on header creation from a summary of text
- Generates valid Bloomberg query lanague code
- Some task it does not do as well as GPT-3 for instance the Reading Comprehension (non finance test). It still does quite well.
