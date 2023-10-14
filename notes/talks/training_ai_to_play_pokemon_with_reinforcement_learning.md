### [Training AI to Play Pokemon with Reinforcement Learning](https://www.youtube.com/watch?v=DcYLT37ImBY)
- Reward based on exploration (new images -> explore)
  - They should have used some nn model for this uniques though, but sounded like they stored it in a dictionary and then compared
  - ^ they actually got stuck in a animation scene also because of the way they implemented this reward check
- Added objective to level up pokemon
  - reward is sum of rewards
- Loose battles = -1 reward
- Reward -13
  - The Agent deposited a Pokemon into the PC causing issues with the level reward as it went down
  - Reward is adjusted to max(level delta, 0)
- The AI has discovered PRNG manipulation to be able to catch a pokemon on the first try
  

### Training
- They used PPO algorithm
- They used PyBy with multiple running instances
- They added 3 image stacked to have some temporal context and also added a small status bar for the model to view


[Code for the project](https://github.com/PWhiddy/PokemonRedExperiments)


