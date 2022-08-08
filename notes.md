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

