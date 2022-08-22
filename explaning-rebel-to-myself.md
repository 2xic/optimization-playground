# Rebel

Rebel is similar to alphazero, expect it works on imperfect-information games.

## Notation

### State
So a state is now expanded on, and now more generally means an probabilistic belief distribution of all agents about what state they may be in.

This is refereed to as PBS in the paper. It's built based on all public observations (actions all agents can see).

### PBS and infostates
So what is a bit confusing in the paper is where the role of the PBS vs infostate.
Based on my understating from page 5, it learns an infostate-value function instead of a PBS value function.
What is the difference ? 
Infostate is the sequence of actions and observations.


## Building blocks
- Policy + value network is used for search

## Search in a depth-limited imperfect-information subgame
- The subgame is solved in the discrete representation and solution is the converted to the belief representation
- CFD outputs a policy profile in the subgame.
- 