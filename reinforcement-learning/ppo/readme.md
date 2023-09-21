# [Proximal Policy Optimization (Spinning up)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
PPO is an on-policy algorithm and can be used for both discrete and continuous action space. 

Which builds on top of [Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html#id4)


## Background 
### Value function
The value function is the value of the state

### Action value function
Expected reward of the action in a state

### Advantage functions
$$A^\pi(s,a) = Q^\pi(s,a)- V^\pi(S,a)$$
The advantage is the on policy action-value function vs on-policy value function.

### [Rewards to go](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-reward-to-go-policy-gradient)
Goes over the reward sum
```python
for i in range(n)[::-1]:
    reward_to_go[i] = reward[i] + rewards[i + 1] # if i + 1 < n else 0
```

## PPO-CLIP objective 
PPO clip objective is the 

$$min(\text{PPO ratio} \cdot Advantage, clip(epsilon, advantage) )$$

### [PPO ratio](https://huggingface.co/blog/deep-rl-ppo#the-ratio-function)
Use the ratio of distribution of the old policy gradient and new policy gradient 

### Clip function
If the advantage is greater than or equal  0 then $(1 + \eps)A$ else $(1 - \eps)A$ 
```
clip(epsilon, advantage)
```

## [OpenAi dota2](https://cdn.openai.com/dota-2.pdf)
They used PPO. 

## documents
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
- [ Policy Gradient Algorithms ](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf
