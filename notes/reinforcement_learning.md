## Reinforcement learning
For the paper notes that are so short that they don't need their own file.

### [Equivariant MuZero](https://arxiv.org/pdf/2302.04798.pdf)
Tries to make MuZero more data efficient by having symmetry be apart of the world model.

# [RL^2 - fast reinforcement learning via slow reinforcement learning](https://arxiv.org/pdf/1611.02779.pdf)
Idea is to learn a reinforcement learning within RNN weights using a (existing) RL method. In other words (and I quote the paper) learning a RL algorithm as a reinforcement learning problem 

## [Go-explore: a new approach for hard-exploration problems](https://arxiv.org/pdf/1901.10995.pdf)
Core idea: Explore new states by revisiting old promising ones

1. Remember state that are previously visited
2. Return to a promising state (no exploration)
3. Start exploring
4. Run imitation learning on the best trajectory 

There are more to the idea, but that is the rough idea.
This method was enough to beat the record for "Montezuma’s Revenge" by 4x the previous state of the art.

## [John Schulman - Reinforcement Learning from Human Feedback: Progress and Challenges](https://www.youtube.com/watch?v=hhiLw5Q_UFg&t=2560s)
- Hallucination connections to Behavior cloning
  - Knowledge graphs 
    -  What the network has learned
    -  It could be different from what the labelers know
    -  Which again can make the model start to hallucinate if the task if the models is then fine-tuned for q&a
     -  Same is true in the opposite direction, networks knows something that the labelers does not know about, and then you have the problem that the network might be trained to say "i don't know" when it does now.
-  Interestingly he thinks that the open source models might make up less information than GPT-4 (and this is because of the fine tuning the way I interpreted him - and I would say that is a good hypothesis). 
- Does the model know ? 
  - Yes, it's trained to minimize the log loss
- Solution (?)
  - Use RL to adjust the output distribution to allow it express uncertainty
- Long form
  - Usually the answer is a mix of right and wrong
  - Ranking of results of RLHF is also difficult because correct answer is very context dependent
  - 
- Retrieval and citing sources
  - Information not in the training data
  - WebGPT
- Shows the ChatGPT plugin with browsing
- Open Problems
  - Expressing uncertainty
    - He mention the idea of having multiple agents compete, I think this might be good approach
  - Going beyond labellers
    - Labellers are like feature engineering, it doesn't scale. Alpha go vs stockfish etc etc.
  - Generating knowledge
    - Prediction / Compression
    - Deduction
  - 

### [Reinforcement Learning for Language Models](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
Post that is somewhat connect to the talk, but not written by the speaker.

So the author here thought originally that RL might not be the way, and that supervised learning could be more useful for aligning models. However, he later came to the conclusion that RL is nice.

Training a LLM usually takes 3 steps. 
- The pre-training phase where the model learns the language representation
- Aligning of the model
  - Supervised learning : Given some input prompt X, do action y. 
  - RL: Model is not given the y, but should generate the best possible answer and receives a reward based on that
RL is harder to do than supervised, one of the reasons is because writing a reward function is hard.

Reasons supervised learning is hard
- It's hard to come up with one correct answer. So the model might be punished when it has produced a answer that is valid, but not the expected one
- RL allows the model to not only learn from representations, but also thinks more "deeply".
- There is also the problem that supervised learning does not necessary know about the models internal state. Some dataset might contain information not in the training of the model so the model will learn a mapping that might make the model more confused.
- Hard to tell the model to say "I don't know" when doing it with supervised learning

### [Neural Optimizer Search with Reinforcement Learning](https://arxiv.org/pdf/1709.07417.pdf)
Use a RNN to control the handling of the gradient (average it for instance). The operations follows a simple pattern described in section 3.1. The idea is to have it simple, but make it able to express the same functions as common optimizer.
Goal for the model is to maximize the models performance. 

The paper is cool, might be worth implementing.
