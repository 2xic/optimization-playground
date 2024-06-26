### [The Alignment Problem from a Deep Learning Perspective](https://arxiv.org/pdf/2209.00626.pdf)
- Authors argue that if models keep being trained like they are today, they could learn to be deceptive and learn gorals outside their learning distribution
- Challenges
  - Reward mis-specification and reward hacking
    - Models learn to exploit bugs in the environments all the time. It's hard to specify a policy. This is why RL with human feedback is useful, but even that has flaws the system can exploit.
    - RL with human feedback is also problematic because it is built on human biases
  - "Situational awareness" 
    - The model might learn what information should be used in a given situation. Knowing this the system can trick the person auditing the system by giving it the answer they want to heard instead of the "right" answer
  - You could also have a perfect reward speciation, but have the system be situations aware and thereby not following the policy when it's put in production
  - 

[Tweet thread from one of the authors](https://twitter.com/RichardMCNgo/status/1603862969276051457)
- Summary of the paper ideas 
- RL models can learn to do reward hacking on human mistakes in feedback. Language models can show similar behavior when they hallucinate. This has been shown in [Building A Virtual Machine inside ChatGPT](https://www.engraved.blog/building-a-virtual-machine-inside/) and [ChatGPT pretends to not know the date](https://twitter.com/goodside/status/1598890043975774208)
- 

### [Introducing Superalignment](https://openai.com/blog/introducing-superalignment)
They want to create a model that acts as an alignment researcher.


[OpenAI Our approach to alignment research](https://openai.com/blog/our-approach-to-alignment-research)
This is the OpenAI "master" plan 

```
1. Training AI systems using human feedback
2. Training AI systems to assist human evaluation
3. Training AI systems to do alignment research
```
^ quoted from the link.

[Healthy critic that I agree with on this post](https://twitter.com/ProfNoahGian/status/1710633967911583994)

### [Weak-to-strong generalization](https://openai.com/research/weak-to-strong-generalization)
[Code is released](https://github.com/openai/weak-to-strong)

[Paper](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)

They use GPT-2 to generate labels for GPT-4 to see hwo the performance becomes. They noticed that the stronger model is able to generalize better than the weaker model. They are still not able to get to full RLHF level performance, but it's better than naive fine tuning.
The reason for having a model do this instead of a human is that in the future - it will just be too complex for a human to be able to fully validate some of the results of the model. We need more automated and robust solutions.

The process looks like this
1. Train the weak model on part of the dataset
2. Train the strong model on some other part of the dataset (without any supervision)
3. Have the weak model generate labels for the second part of the dataset for the new model and then train the strong model on that
4. 
  