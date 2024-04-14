**not very polished, some raw thoughts**
**todo:: add years of various papers and make it a real timeline**

It's very hard to reason about some of the accomplishments in machine learning in a retrospective context, but I still want to collect some of my thoughts on this.

Sadly a lot of today is just hype, we have come a far way, but I don't want to buy into the hype and looking at some retrospect help with that.

### Chess = AI
So in the middle/late nineteens, [Deep blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)) was being developed and it beat Garry Kasparov in 1997. How ? Search! It used [alpha-beta](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)#Hardware) search algorithm to search the game space and evaluate the best game positions.

DeepMind published this year a paper ["Grandmaster-Level Chess Without Search"](https://arxiv.org/pdf/2402.04494.pdf), but still isn't super human without search. There is also some [skepticism of their claims](https://www.lesswrong.com/posts/PXRi9FMrJjyBcEA3r/skepticism-about-deepmind-s-grandmaster-level-chess-without).

### Go = AI
Chess has a large state space, but the state space of Go is much larger. Using the same methods that beat Garry Kasparov in a model for Go won't do very well because the search space is so large.
In the middle of the twenty-tens AlphaGo project was started within Deepmind. They wanted to evaluate deep learning methods on the game of Go.
In 2016 they reached a great milestone by beating the then world champion [Lee Sedol](https://en.wikipedia.org/wiki/Lee_Sedol#Match_against_AlphaGo). He later famously [retired from professional play](https://en.wikipedia.org/wiki/Lee_Sedol#Retirement_from_professional_play).
How was this done ? Classical search techniques with some deep learning on top. The search was done by monte carlo search methods, but the value function was done by a deep learning model.

They have done some further research and later developed methods like [MuZero](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/), but they all still use classical search methods for the planner. 

**[There still isn't a raw NN that beats human at Go](https://twitter.com/polynoamial/status/1501534834950213632)**. 

### Poker = AI
[Noam Brown](https://noambrown.github.io/) who tweeted the tweet I cited above on raw NN and Go was one major driver for the first ai that [won](https://qz.com/907896/how-poker-playing-ai-libratus-is-learning-to-negotiate-better-than-any-human) the no-limit poker.
You know the drill, how does it work ? [Search](https://knowen-production.s3.amazonaws.com/uploads/attachment/file/3485/AI%2Bfor%2Bmultiplayer%2Bpoker.pdf)!

### Jeopardy == AI
**todo: add a section on jeopardy**

### Dota2 (2018) = AI = Starcraft (2019)
Games with that aren't perfect information games = AI ?
- [OpenAi Five](https://github.com/alexis-jacq/Pytorch-DPPO)
  - Disclaimer - there was some limited heros and items to select from here[1][2][3].
  - The model itself is very simple LSTM + PPO.
- [AlphaStar](https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
  - [The unexpected difficulty of comparing AlphaStar to humans](https://aiimpacts.org/the-unexpected-difficulty-of-comparing-alphastar-to-humans/)

[1] https://old.reddit.com/r/DotA2/comments/94udao/team_human_vs_openai_five_match_discussions/e3p0nx7/
[2] https://old.reddit.com/r/reinforcementlearning/comments/8tqzvq/openai_dota_update_ppo_lstm_reaches_amateurlevel/
[3] https://jacobbuckman.com/2018-08-06-openaifive-takeaways/


### Evolution of GANs, why are Gan better?
[4.5 years of GAN progress in one image](https://blog.acolyer.org/wp-content/uploads/2019/02/gans-tweet-1.jpeg?w=480) so image generations are clearly something the models are getting better at, but how ? 
Compute, "smarter" algorithms (smarter is up for debate here, we could also just call it clever tricks), architecture changes.

### Evolutions of YOLO
The [history of Yolo](https://deci.ai/blog/history-yolo-object-detection-models-from-yolov1-yolov8/)
1. [The first version of Yolo](https://arxiv.org/pdf/1506.02640v5.pdf) had a unique approach to how to go about solving the detection problem. It would make the image into a grid, then predict each grid class score and this would then be aggregated with [s-nms](https://arxiv.org/pdf/1704.04503.pdf). 
2. [The second version of Yolo](https://arxiv.org/pdf/1612.08242v1.pdf). They added batch normalization to the model, improved how they train the classifier to use higher resolution inputs, [anchor boxes](https://www.thinkautonomous.ai/blog/anchor-boxes/)
3. [The third version of Yolo](https://arxiv.org/pdf/1804.02767.pdf). It's a less fancy update, but uses 
4. [The forth version of Yolo](https://arxiv.org/pdf/2004.10934.pdf), the lead developer Joseph Redmon [steps away](https://twitter.com/pjreddie/status/1230524770350817280) from the project. [Bag-of-Freebies](https://medium.com/visionwizard/yolov4-bag-of-freebies-dc126623fc2d) is added (augmentation, data imbalance / bias ), [bag-of-specials](https://medium.com/visionwizard/yolov4-version-2-bag-of-specials-fab1032b7fa0) is added (changes in activations, DenseNet)

Then there was some controversy, but you get the idea. The general idea didn't evolve much, but we did [some alchemy](https://www.youtube.com/watch?v=x7psGHgatGM) and the model still improves some.

### Evolutions of AlphaZero
1. [AlphaGo](https://www.davidsilver.uk/wp-content/uploads/2020/03/unformatted_final_mastering_go.pdf)
- MCTS
- NN for the state evaluation
2. [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf)
- +Self-play
3. [MuZero](https://arxiv.org/pdf/1911.08265.pdf)
- + Improved search by the internal state representation

### Evolution of Siri
**todo**

### Evolutions of  GPT
Most (non machine learning) people don't realize GPT was first released in [2018](https://en.wikipedia.org/wiki/GPT-1).
1. [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Train model on a lot of data
- Predict next token
- Fine-tune
2. [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- Layer normalization
- More data, larger vocab and context size
- You start to see the emergence of something. The model is able to answer questions about text etc.
- The model does well on multiple diverse tasks.
3. [GPT-3](https://arxiv.org/pdf/2005.14165v4.pdf)
- "few shot learners" -> model is able to learn based on context.
- [RLHF](https://openai.com/research/instruction-following)
4. [GPT-4](https://paperswithcode.com/method/gpt-4)
- Who knows what is in this.
- [Mixture of experts](https://www.youtube.com/watch?v=WJWHIZoBOj8).
5. GPT-5
- Who knows what's gonna happen, but we can see that websites starts blocking the [crawler](https://twitter.com/AznWeng/status/1777688628308681000). There is less good data to train on. RLHF doesn't scale.

What is this ? Hard to say, some say it's the recipe for AGI, but it clearly lacks a planner.

### The era of LLMs
- ["AI software engineer"](https://www.cognition-labs.com/introducing-devin) 
  - https://medium.com/@avra42/is-databutton-the-new-full-stack-ai-alternative-to-devin-for-app-development-888a8e33a54a
  - https://twitter.com/batwood011/status/1767722399917813951
  - https://www.youtube.com/watch?v=tNmgmwEtoWE
  - https://news.ycombinator.com/item?id=40008109
  - 
- [AGI in 5 years](https://www.reuters.com/technology/nvidia-ceo-says-ai-could-pass-human-tests-five-years-2024-03-01/) ?

My personal take on all of this is that LLM are great at compressing information, but not much more. They hallucinate so even the retrieval of information is sometimes very flaky.

## Aside: compute
**todo: add note on moore's law**

*numbers are likely off by magnitudes, but might still be worth thinking about*

["a new dominant species"](https://youtu.be/jem1unAG4_k?feature=shared&t=10721). Synapse and neurons. Multiply (synapses) and accumulate (neurons). `bioFlops = synapses * clock rate`. 
An ant (2 GFLOPS) ~ raspberry pi. Bee (200 GFLOPS) ~ 500 ants ~ Comma 2 device (smartphone). Mouse ~ 100 TFLOPS ~ Multiple GeForces. Cat ~ 2 PFLOPS ~ one rack. One human ~ 20 PFLOPS ~ one rack of TPU v3. 
Humanity is a network. Cats are not networked, dogs are on LAN, but only humans are networked like the Internet. Collective intelligence. The brain software might not be particular good given it's hardware. 
- The human race is about 1.6 (10^26) FLOPS = 100 billion petaFLOPS
- Silicon flops in the world is about 10^21 (unsure where he got the numbers from)
- Factor of 10^5 off.
Silicon flops are continuously evolving, biological flops are not. If things keep evolving like this we will no longer be the dominant species. 

We have all of this compute, but our machine learning doesn't match it. We clearly have good hardware, but the software (the algorithms we have) clearly doesn't match.

OpenAI has a interesting plot between [AI and compute](https://openai.com/research/ai-and-compute). 

## Side note: Google [Duplex](https://research.google/blog/google-duplex-an-ai-system-for-accomplishing-real-world-tasks-over-the-phone/)
What ever happened to Google Duplex ? Google had this amazing demo at [Google IO 2018](https://www.youtube.com/watch?v=D5VN56jQMWM). It turned out to be a [scam](https://old.reddit.com/r/singularity/comments/mzadng/what_happened_to_google_duplex_ai_assistant_used/).

With todays technology something like duplex is likely something that can be built. LLMs have some good internal state representation and we have TTS 


### E2E is the way
So we have seen some great accomplishments over the years. We see time and time again that learning things e2e is the way.
- [AlphaDev](https://deepmind.google/discover/blog/alphadev-discovers-faster-sorting-algorithms/)
- [Taco bell comma.ai trip](https://www.youtube.com/watch?v=cAWScxZuc0g) 
- 

 
### Side note: Self-driving cards
Elon Musk is famous for his [self-driving cars predictions](https://www.consumerreports.org/cars/autonomous-driving/timeline-of-tesla-self-driving-aspirations-a9686689375/) having promised it will only be two years away for a decade soon.
Robot taxis have also been promised for almost [half a decade](https://twitter.com/elonmusk/status/1148070210412265473) to just be a few years away. [Ali Rahimi](https://youtu.be/x7psGHgatGM?feature=shared&t=683) also mentioned this during his Alchemy talk.

So what are the methods people use today ? There are a few various approaches to this problem (most of them are [scams](https://www.youtube.com/watch?v=w2Ok7jUUB9U))
- [comma.ai](https://blog.comma.ai/) e2e machine learning is the way. People can buy and use the product. 
- Telsa auotpilot is doing [a lot](https://www.youtube.com/watch?v=j0z4FweCy4M).
- Waymo ? 
- Cruise ?

They all still have problems with their planner. Credit where credit is due - both Comma.ai and Tesla are having some impressive videos made by users on Youtube.
- [Comma.ai taco bell](https://www.youtube.com/watch?v=SUIZYzxtMQs) 
- 

[We actually had "self-driving" cars in 1989](https://www.theverge.com/2016/11/27/13752344/alvinn-self-driving-car-1989-cmu-navlab). Why hasn't things gotten further ? 
Because the algorithms lack planning.

### Robotics
So we are seeing some cool stuff here
- [1x](https://www.1x.tech/) has some impressive demos and are [supposedly](https://www.1x.tech/discover/all-neural-networks-all-autonomous-all-1x-speed) only using neural networks and video at 1x speed. Sadly no buy button so probably some hype here also.
  - (to confirm)They probably use LLM for the planning.
  - 
- [bostondynamics](https://bostondynamics.com/) does a lot of [classical stuff](https://old.reddit.com/r/robotics/comments/kp35lz/boston_dynamics_how_do_they_do_it/)
- 

### The problem today: planning
So from we can gather from above all current solutions have one giant problem - they can't do long term planning.
