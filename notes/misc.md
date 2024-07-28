## [Learning skillfull medium-range global weather forecasting](file:///Users/brage/Downloads/science.adi2336.pdf)
[Source code](https://github.com/google-deepmind/graphcast)

- Improved machine learning model by using historical data
  -  Allegedly based on what is said in teh paper - traditional methods uses more compute resources for greater predictions 
  -  https://en.wikipedia.org/wiki/Numerical_weather_prediction
-  Current methods often uses supercomputers while GraphCast can use a single Google TPU v4
-  The model itself 
   - Autoregressive -> Can be rolled out based on its own inputs
   - Input is two two weather states and predicts the 
-  Trained for 4 weeks with 32 Google TPU v4

## [Deep Reinforcement Fuzzing](https://arxiv.org/abs/1801.04589)
Uses Q-learning as a way to do fuzzing. There isn't much more to say, but the idea is quite cool as generally fuzzing algorithms use something like evolutionary algorithms.

## [Revisiting Neural Program Smoothing for Fuzzing](https://dl.acm.org/doi/pdf/10.1145/3611643.3616308)
They evaluate ML based program fuzzers
- They conclude that the promises of Neural based program fuzzing does not hold
- They created a platform [MLFuzz](https://github.com/boschresearch/mlfuzz) to evaluate new Ml fuzzers
- Traditional methods beat ML based

### [Deep Networks Always Grok and Here is Why](https://arxiv.org/pdf/2402.15555.pdf)
- Studying grooking empirically
- Observations point at grooking is not limited to certain tasks or architectures
- 

### [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/pdf/2403.02545.pdf)
- They build a model architecture that allegedly is better at scaling for recommendations tasks
- They make some reusable block which is scalable
- Seems to scale a lot better than other models with same parameters count

## [Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks](https://arxiv.org/pdf/2403.05185.pdf)
They made some data observations which was kinda intresting
1. Audiobook streams are mostly dominated by power users and popular titles
2. "Podcasts user tastes and content information are informative for inferring users’ audiobook consumption patterns."
3. Accounting for podcast interactions with audiobooks is essential for better understanding user preferences.
- They use Graph based neural networks and seem to combine both the podcast and audiboook data part of the model. The system generates daily user and audio-book features. 

### [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)
- Created a dataset Abstraction and Reasoning Corpus ([ARC](https://github.com/fchollet/ARC)) to better model intelligence in a ML model.
- There was a [Kaggle competition](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/leaderboard) back on the day for this also

### [The Hardware Lottery](https://arxiv.org/pdf/2009.06489)
- Hardware is a driving factor for whether a research idea is successful or not.
  - The idea for Neural networks is more than 30 years old, but it's just in recent years it had a breakthrough because of the hardware limitations 30 years ago.
- Hardware lottery therefore refer to whether a research idea is successful because of current hardware capabilities or not.
  - We want the best algorithms to win because they are best, not because of some lottery.
- 

### [The Platonic Representation Hypothesis](https://arxiv.org/pdf/2405.07987)
- TLDR: Representation for vision and text models are converging
  - This seems to happen also when trained with different objectives and data

## [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- Silver medal winner in the international math olympiad :D
- RL Based with LM and uses the Lean proof engine.
- Eval
  - Solved one question in minutes
  - Spent up to 3 days to solve the others
  - They scored 28 / 42 points
- AlphaGeometry 2
  - Upgraded version of [AlphaGeometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)
  - neural-symbolic with a language model based on Gemini. It uses search trees to tackle more complex problems.

