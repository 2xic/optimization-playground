# Rebel

Rebel is similar to alphazero, expect it works on imperfect-information games.

## Notation

### Discrete representation 
How the game is actually played. Without an middleman (as described in the section 4).

### Belief representation
The game representation with an middleman.

### Expected value
Expected value is the sum of all future rewards.

### State
So a state is now expanded on, and now more generally means an probabilistic belief distribution of all agents about what state they may be in.

This is refereed to as PBS in the paper. It's built based on all public observations (actions all agents can see).

### PBS and infostates
So what is a bit confusing in the paper is where the role of the PBS vs infostate.
Based on my understating from page 5, it learns an infostate-value function instead of a PBS value function.
What is the difference ? 
Infostate is the sequence of actions and observations.


#### PBS in a perfect information game
- S_i(S_pub) is the information state that player i can play in a publlic state (S_pub)
  - p(S_i(S_pub)) is the probability distribution over the given info state
    - then PBS = (p(S_1(S_pub)), p(S_2(S_pub)), ... p(S_N(S_pub))) where N is the agents count
- 

## Building blocks
- Policy + value network is used for search

## Search in a depth-limited imperfect-information subgame
- The subgame is solved in the discrete representation and solution is the converted to the belief representation
- CFD outputs a policy profile in the subgame.
- 