# [Building AI Trading Systems](https://dennybritz.com/posts/building-ai-trading-systems/)

This blog post cover a lot of interesting topics of deploying ML in a more competitive and "unique" environment.

## Reward are great labels
- How would you train a supervised system in the context of the stock market ? 
  - Seems tricky right
- The author also thinks reward is the way as it is easy to define (profits = reward) and train with. That said this doesn't mean applying RL to the problem automatically makes it easy
  - Rewards are sparse
  - Requires generalization

#### Simulation is the way
- Given the environment being a orderbook / similar it can be modeled and the agent can be trained in a simulator
  - Which is also "required" for the model to be resilient against api issues, slippage etc. 
  - ^ how do you train a supervised system to handle that ? 


