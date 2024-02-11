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

### [Are Emergent Abilities of Large Language Models Mirage?](https://arxiv.org/pdf/2304.15004.pdf)
My timeline is full of claims that big models have "emergent abilities", but theses authors claim that it might not be the case, and that the claims are the results of benchmarks/metrics not being measured good enough.
"Emergent abilities" is defined as capabilities that big models have that smaller models don't have.
Linear plots seems to work well and gives a more predictable view, but most authors seems to use nonlinear.
Section five is funny, they drive the point home by adding "emergent abilities" to a mnist LeNet vision model.

[twitter thread](https://twitter.com/andrewlampinen/status/1652962934107127808?s=12&utm_source=pocket_saves) -> 

### Various ChatGpt notes
- [it can play tic tac toe](https://twitter.com/michael_nielsen/status/1649848364232773633?utm_source=pocket_saves)

### [Blockchain Large Language Models](https://arxiv.org/pdf/2304.12749.pdf)
Given the amounts of attacks on DeFi applications, getting a real-time detection system (IDS) can be very useful. State of the art application for these kind of tasks usually use reward based methods, or custom heuristics methods.

Key insight from the authors "Attacking transaction usually have a execution path is very different from other execution paths" based on this it should be possible for a model to learn the representation needed to detect abnormal transactions.

The system seems to consider both the public mempool, but also the txs that are sent to a private mempool and later finalized (where BlockGPT only can see the finalization). 

The system works by having a transaction tracer that is able to see the function calls, input / output data, and see the execution path. The model is trained on a historical dataset of transactions. An detection model is used to score/ rank the trace of any new transactions.

"Intermediate Trace Representation" -> an tree structured trace is used as input into the model (see listing 1 for example, and page 4 for additional details)
Each node in the tree is one hot encoded into a vector based on the grammar. Positional encoding + transformer encoder allows the trace to be parsed in sequences.
The loss seems to be as simple as predict the next token (or to be more precise the probability distribution). 


-> Dataset is from a previous study. Which one ? 
  -> Sounds like it's from https://arxiv.org/pdf/2208.13035.pdf but haven't found a source there

### [PaLM 2 Technical Report](https://arxiv.org/pdf/2305.10403.pdf)
-> Transformer architecture
-> Upgraded (and improved) version of [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) (released last year)
-> Side note: I have seen discussions on [Reddit](https://www.reddit.com/r/OpenAI/comments/13gx1g1/palm_2_vs_gpt4_why_google_is_having_a_hard_time/) and Twitter that both indicate that PaLM is not as good as GPT4.
  - In the paper they have some evaluations where GPT4 also is tested, and PaLM does beat GPT4 at some benchmarks

What they did to improve the model
-> Used scaling laws, and used compute optimal scaling of dataset + compute
-> Multilingual dataset
-> Switched to a transformer architecture (interesting at previous model this not use this)

Very little details about the model is shared, and most of the paper is about benchmarking and evaluating the capabilities of the models.

### [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442.pdf)
This is interesting, they have a town of LLMs Agents, and seed one of the agent with the task of hosting a Valentine Day party. The result is the agents starts to work together and get to know each other to be able to fullfil the plan.

The components used
-> Memory stream = for long term memory
-> Reflection = allows the model to draw conclusion about itself and the other agents
-> Planning = creating actionable items from the reflection

Quite interesting!

### [An adventure in unleashing large language models (LLMs) on tabular Kaggle competitions](https://arize.com/blog-course/applying-large-language-models-to-tabular-data/)
Can we use LLM to predict tabular datasets ? (as we know Deep Learning have had difficulties with this for a long time)

With zero background knowledge, zero data cleaning and zero feature development they were able to use an LLM to beat the median score on the leaderboard. 

### [Large Language Models are Built-in Autoregressive Search Engines](https://arxiv.org/pdf/2305.09612.pdf)
LLMs are able to help with information retrieval
The architecture seems to be 
-> Input urls + query
-> Document text is fed into the model
-> Model scores document based on query

The interesting part is that the urls are generated by the language model. Tested this myself, and it seems to work well.

### ["Large Language Models trained on code reason better, even on benchmarks that have nothing to do with code."](https://twitter.com/hardmaru/status/1657906920257372160?s=12&utm_source=pocket_saves)
Contains link to some great discussion. For instance [this](https://old.reddit.com/r/MachineLearning/comments/13gk5da/r_large_language_models_trained_on_code_reason/jk29amd/) comment which mentions a few interesting things (most of it seems to be from [this](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1) document on notion). It seems that concepts like "chain of thought" is the result of models being trained on code. To back this claim up, initial GPT-3 was not trained on code and could not do chain of thought, other models like Palm which has some code in the dataset can do chain of thought. 

### [ State of GPT | BRK216HFS ](https://www.youtube.com/watch?v=bZQun8)
Training pipeline
1. Pre-training (base model)
   1. 99 % of the compute time
   2. Big dataset + long tranining
   3. Tokenization (Byte pair encoding)
   4. The final model here loves to complete documents, but not much more.
2. Supervised finetuning 
   1. Change up tranining dataset to a smaller one, but with high quality data
3. Reward modeling
   1. The transformer will give a "guess" for the reward for each completion
   2. ^ this is how you can enforce the reward model
4. Reinforcement learning
   1. RL with respect to the reward model

Model size
- ~ 30-50 000 vocab
- ~2048 context size
- 100+ B ~ model parameters
- Cost a few millions to train

Why RLHF ? 
- It's easier to compare stuff (RL) then create stuff (supervised).

- The RLHF model seems to give less entropy output than the base model.

---

Karpathy also referenced [Large Language Model Guided Tree-of-Thought](https://arxiv.org/pdf/2305.08291.pdf)


## [Large Language Model Guided Tree-of-Thought](https://arxiv.org/pdf/2305.08291.pdf)
Problems with LLMs today
- Lack of correct checking
- No backtracking (so if we get a bad token, there will be a bad path)

The idea to solve these problems is with a tree search. The setup has a few components prompt agent, checker module, memory module and ToT controller. The idea is first to make the LLM generate first step of a solution, and have the checker module check if the step is reasonable before generating the next step. The ToT controller checks that the final step also makes sense, or if it should backtrack and explore other options. The memory module keeps track of the conversation, and other useful hints for the LLM.

There is also attached psuedo code for the algorithms, and the benchmark results for psuedo gives it SOTA for LLMs.

### [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/pdf/2305.16291.pdf)
[tweet](https://twitter.com/DrJimFan/status/1662115266933972993)
[karpathy tweet](https://twitter.com/karpathy/status/1662160997451431936?s=12)

(Skimming through this paper)

Using an LLM to explore Minecraft worlds, and beats SOTA for various exploration metrics.


Building blocks
- Automatic curriculum that maximizes exploration.
  - Gives you idea for what to do next (given a wooden pixaxe, try to craft a stone pixaxe etc)
- Skill library for storing and retrieving complex behaviors.
- Iterative prompting mechanism that incorporates environment feedback for program improvement.


Notes
- I don't see it compared against [VPT](https://arxiv.org/pdf/2206.11795.pdf). 
- Looks like the model actually writes code that is executed (I guess that makes sense, but it changes how the environment is viewed compared to other existing solutions)
- **Self-verification** is the most important component which by having another GPT-4 act as a critic of the agent. 

### [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/pdf/2305.15717.pdf)
TLDR; Imitation models make answer that seems very good, but in reality it will fail at factuality. Instead they will learn the style of ChatGPT like answers.

Imitation model = Weaker model learns from stronger model.


### [Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks](https://arxiv.org/pdf/2305.14201.pdf)
Found this by this [tweet](https://twitter.com/rasbt/status/1661754946625105920?s=12). They point out what I assume is the key here the fact that the tokenization is split by each digit. 

Looking at the results, it still does not achieve 100% on all the benchmarks, but beats GPT-4 quite a bit (especially on multiplication).

### [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://crfm.stanford.edu/2023/05/22/alpaca-farm.html)
TLDR; 
-> Use LLMs to create a "simulation" of human feedback to bootstrap model
-> Train on real human feedback after model is bootstrapped.

### [OpenAI Watch](https://openaiwatch.com/)
Interesting idea, track model capability by it's ability to draw unicorns.

### [LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206.pdf)
- Trains a LLM without RLHF, and instead use supervised prompt -> response dataset
- Does not do quite as well as GPT-4 in a human "preference" study, but does about 50% as good.
  - That said the dataset is just 1 000 prompt -> response

### [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264v1.pdf)
- Unique data > Training on same data
- More repetitions of the data -> The value of adding more compute becomes less
  - That makes sense, doesn't it ?
- 

They are studying scaling behavior in data constrained environments. From the figures
-> More epochs hurts the model on a fixed data budget 
-> More data = bigger loss, more data is good
-> Multiple epochs with repeating data = good

[tweet](https://twitter.com/blancheminerva/status/1664097432496488450?s=12)

[tweet ](https://twitter.com/muennighoff/status/1661895337248686081?s=12)


### [Let’s Verify Step by Step](https://arxiv.org/pdf/2305.20050.pdf)
[tweet](https://twitter.com/8teapi/status/1664123104074022917?s=12)

OpenAi created a large [dataset](https://github.com/openai/prm800k) of step-level labels, and trained a model on that.
That model is then used to grade another models reasoning steps.

This is much better than just an outcome based model.

### [RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/pdf/2307.04349.pdf)
Use unit tests to guide an LLM to write code. 


### [How Is ChatGPT’s Behavior Changing over Time?](https://arxiv.org/pdf/2307.09009.pdf)
TLDR: GPT behavior has drastically changed over the last couple of months. Mostly for the worse.

[related tweet thread](https://twitter.com/svpino/status/1681614284613099520)

[OpenAi even extended the support for the old models because users were not happy](https://twitter.com/openai/status/1682059830499082240?s=12)

[Counter article](https://www.aisnakeoil.com/p/is-gpt-4-getting-worse-over-time) that adds a bit more nuance to the issue. 

### [Llama 2](https://ai.meta.com/llama/)
- Public weights
- [Karpathy therad on llama 2](https://twitter.com/karpathy/status/1681667444895539202?s=12)

[Difference between llama 1](https://www.apps4rent.com/blog/llama-vs-llama-2/)
- More data and larger context window
- Uses RLHF
- 

### [The Internal State of an LLM Knows When its Lying](https://arxiv.org/abs/2304.13734)
- They train a classifier based on the the hidden layers of a LLM 
- Using the classifier they are able to more robustly know if a statement is true or not
- This also shows that the LLM is aware of the fact that it's lying.

## [prompt-optimizer](https://github.com/vaibkumr/prompt-optimizer)
Just use less words to save money on the llm apis.

It is just a wrapper around `nltk`.

## [HYPOTHESIS SEARCH: INDUCTIVE REASONING WITH LANGUAGE MODELS](https://arxiv.org/pdf/2309.05660.pdf)
- Give the LLM some program examples
- Have it generate a program for the examples
  - If it succeeds on all examples then return it
  - If it fails, have `n` searches where you give it the feedback (exceptions etc) so it can retry

[Twitter thread](https://twitter.com/ericzelikman/status/1701405044610851230)

### [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/abs/2310.12931)
[Tweet thread](https://twitter.com/DrJimFan/status/1715401002957013076)

Uses LLMs to do reward engineering. Quite cool. It uses "reward reflection" to generate new reward functions.

My immediate reaction by looking over the paper is that I'm unsure if this somehow could be part of the training dataset or not. The changes applied by the model 
- Just searching on github for some of the changes does not reveal anything [obvious](https://github.com/search?q=0.5+*+rotation_reward+%2B+0.3+*+distance_reward+language%3APython&type=code) and [here](https://www.google.com/search?q=%220.5+*+rotation_reward%22+site%3Agithub.com&client=firefox-b-d&sca_esv=576194101&biw=1233&bih=886&ei=sx04ZfKrFviqxc8PqYes0As&ved=0ahUKEwiy0ciZuI-CAxV4VfEDHakDC7oQ4dUDCA8&uact=5&oq=%220.5+*+rotation_reward%22+site%3Agithub.com&gs_lp=Egxnd3Mtd2l6LXNlcnAiJyIwLjUgKiByb3RhdGlvbl9yZXdhcmQiIHNpdGU6Z2l0aHViLmNvbUj9BVDXBFjXBHABeACQAQCYAVegAVeqAQExuAEDyAEA-AEB4gMEGAEgQYgGAQ&sclient=gws-wiz-serp)


### [GPT-4 vision is susceptible to optical illusions](https://twitter.com/fabianstelzer/status/1717131235644875024)
There is some back and forth in the discussion, but seems like it is susceptible to optical illusions.

### [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/pdf/2311.17035.pdf)
- Core results here - Authors show that you can make hte LLM model diverge by having it repeat words "indefinitely"


### [Gemini](https://blog.google/technology/ai/google-gemini-ai/?utm_source=twitter&utm_medium=social&utm_campaign=GDMGemini)
- Google latest big and great model
- Multimodal and beats GPT4 on a rang of reported metrics
  - Looking at some of the videos this looks quite cool, but this seems to have been overhyped marketing.
  - They wrote up a [blogpost](https://developers.googleblog.com/2023/12/how-its-made-gemini-multimodal-prompting.html) describing what they did. 
  - Someone [recreated it with GPT](https://sagittarius.greg.technology/) -> Not smooth ofc, but still cool.
  - [Tweet](https://twitter.com/realGeorgeHotz/status/1733178571999625700)
- [Keypoitns](https://twitter.com/srush_nlp/status/1732427569352323401)
- [AlphaCode 2](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)
- [Highlight thread by Jeff Dean](https://twitter.com/JeffDean/status/1732415881316319667)
- [Technical paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)


## [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b/)
[Code](https://github.com/mistralai/mistral-src)

## [Mixtral 8x7B](https://github.com/ml-explore/mlx-examples/tree/main/mixtral)
[BlogPost](https://mistral.ai/news/mixtral-of-experts/)



### [THE CURSE OF RECURSION: TRAINING ON GENERATED DATA MAKES MODELS FORGET](https://arxiv.org/pdf/2305.17493.pdf)
- Training on model output make it forget - the authors call this "model collapse"

### [https://chat.lmsys.org/](https://chat.lmsys.org/)
Chatbot tournament with leaderboard - cool

### [Offering a tip gives longer responses](https://twitter.com/voooooogel/status/1730726744314069190)
Offer ChatGPT a tip and oyu will get a longer response.


### [LLaMA with block expansion]()
[Paper](https://huggingface.co/papers/2401.02415)

[Github](https://github.com/TencentARC/LLaMA-Pro)

[Tweet](https://twitter.com/shxf0072/status/1743152815961493870)

TLDR: Freeze the old model and add a new block = You can train model without forgetting

### [Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494.pdf)
High level summary
- ELO is ~ 2895
  - ELO of stockfish is ~3632
- They try to use LLM like architecture for training the engine
- Using Stockfish as a source to generate the state values

People on twitter are hyping this up and I think it's cool, but maybe a bit overhyped. Here is a [good take](https://gist.github.com/yoavg/8b98bbd70eb187cf1852b3485b8cda4f) on the paper.


### [Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/pdf/2312.11514.pdf)
- Paper from Apple to limit the latency of running LLMs by reducing the load time of loading from flash to DRAM and allowing larger models to run on machines with limited DRAM
  - The results look quite impressing they can models 2x the size of available DRAM and its load time is 20-25x the inference speed
- How do they do this ? 
  - They don't load the entire model into memory at once.

It has been shown that LLMs have sparisty in the feed forward layers and the authors of the paper leverages this and only load parameters that have a non-zero value or are predicted to have a non-zero value (building upon [Deja vu](https://arxiv.org/pdf/2310.17157.pdf)).

