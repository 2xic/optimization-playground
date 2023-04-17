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
