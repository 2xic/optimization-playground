# [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf)
To make RL scale we cannot handcode a reward function. Curiosity is one method that aims to be a more universal reward function.
To prevent leakage of the true reward function they do not do anything special on the done signal (for instance some people set reward = 0 on done if goal is to maximize score)

The paper conclusion/discussion is that curiosity reward makes the agent learn useful skills.
The one caveat is the noisy-tv problem which is a challenge for this method.
