### Just some notes

- State = Public belief state (PBS)
  - Belief distribution over states
  - Determined by the public observation shared by all agents and the policies of agents


### Notation and background
- Game with N agents
- World state (w) is a state in the game (W)
- A_i(w) -> legal actions for agent i at state w
- T = transition function
  - T(w,a) -> next world state given an action
  - R_i(w, a) -> Reward for agent given action in world state
  -  Given an transition
     -  agent i receives an private observation form function o_priv(i)(w, a, w')
     -  all agents receives an observation o_pub(w, a, w')
        -  All betting actions are public
-  history (trajectory) is a sequence of legal actions
   -  h = (w_0, a_0, w_1, a_1)
-  Infostate (action - observation history or AOH) for agent i is a sequence of an agents observations and actions s_i = (O_i_0, a_i_0, O_i_1, ... O_i_t)
- Public state is a seqence s_pub of public observations



- Agents policy maps an info state to probability distribution. Policy profilers are a tuple of policies.
- The expected sum of future rewards (also called EV) for agent i at history h when all agents play the policy profile pi is denoted v_i^pi(h)
- pi_i^*  is nash equilibrium if it's the optimal policy for all agents.

- Subgames is defined by a root history h
  - It's identical to the original game, but it starts from some given offset h
- depth limtied subgame is a subgame with a max depth into the future

### From world state to public belief network
- Example of 52 cards
  - Without a referee is called discrete representation
  - With a referee is called belief representation.
  - 


----
Some notes from video https://www.youtube.com/watch?v=BhUWvQmLzSk

- Value of a node is dependent on a policy, and not a state.
  - Unlike alphazero
- World state 
  - Public info + Private info
  - Nobody knows this fully
- Infostate is like the history, but conditional on what an agent knows
  - Observations + actions
- 

- Traning
  - Solves the subgame with CFR
  - 

----
Some notes from video https://www.youtube.com/watch?v=mCldyXOYNok
- Humble author! Very good!
- Perfect-information games
  - Value of state 
    - f_white(game state) = value of state for player white in chess
    - INput is state, and output is estimate of the state value
  - Where does the value network come form ?
    - Handcrafted
    - Training on expert
    - Self-play
- Search
  - Allow to solve a subgame
    - Only look 10 moves ahead i.e
    - Estimate the value at the leaf nodes
    - "backprop" the value 
    - play until end of game, train on the tree
  - Thanks to random exploration it should be able to get a good converge of the true value of a state
- Assume that the policy is known
  - v(ROck) is not well-defined
  - V([0.8, 0.1, 0.1]) = -0.6
    - Core idea of ReBel
-   Simle poker game
    -   Each player get a private card
        -   Fold,
        -   Bet
    -   With the referee
        -  Takes a probability distribution of each action from the player
           -  Then outputs the actual results
              -  I bet with 50 % if I have 2
              -  if I have a 3 I fold with 100%
              -  If I have a A I bet with 100%
           -  Even if the referee output action BET
           -  The player still has to update it's internal info state.
              -  It does not know the card
                 -  But it does know that it's not a 3!
              -  Both player knows this information though!
     - Both games reveal information actually...
       - All actions reveal policy information 
- Bayes rule is used to update the distribution of the public belief state
- Public belief states
  - Identical to perfect info states in perfect-info games
- Always have a single unique value 
- Action / state space is continuous and high dimensional
- BUT the action space is convex optimization problem
  - Can use gradient-decent
- 

- ReBel from start to finish :)
  - Whenever an agent acts, generate a subgame and solve it
    - Solve it using fictitious play or CFR
    - Take next action
    - FP / CFR are iterative, so value net must be accurate on every iteration
    - To ensure proper exploration, we stop FP / CFR on a random iteration
  - In a new subgame - Repeat until end of game
    - Final value is used a training examples for all encountered PBSs
    - Use some epsilon- exploration
  - Playing nash at test time
    - Normal players does not reveal policy
      - And what if the other player knows our policy ? 28:24 in video
      - We need to be unpredictable
        - stop CFR at random iteration and assume beliefs from that iteration
    - 

## Notes from blogpost 
- https://towardsdatascience.com/understanding-facebooks-rebel-a-noteworthy-step-in-artificial-general-intelligence-57740917d92f
  - Okay based on this I initialize the pbs to uniform
  - 

- https://ai.facebook.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/
  - "This is why public belief states, or PBSs, are crucial. We need to change the definition of a state so that it is not defined simply by a sequence of actions but also includes the probability that different sequences of actions occurred."
  - 










