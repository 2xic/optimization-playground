[Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf)

- Models such as RNN build complicated transformers by computing a sequence of transformations to a hidden state
  - h_{t + 1} = h_t + f(h_t, \theta)
- What happens if we instead try
  - d h(t) / = f(h(t), t, \theta)
    dt
  - i.e an ode equation
  - From the input layer h(0), and having the solution in H(T)
    - Supposedly this can be solved by an black-box differential equation solver

Benefits of ODE approach
- No need to store intermediate quantities of the forward pass
  - So you can train models with constant memory!
- Can handle data appearing at any time and not just in sequences

----

Gradient computation
- ODE solver is threated as a black box
- Adjoint sensitivity method is used to find the gradients
  - https://towardsdatascience.com/the-story-of-adjoint-sensitivity-method-from-meteorology-906ab2796c73
  - It's also expanded on in the appendix
  - They also include a autograd implementation in the appendix
- Equation 3, 4, 5
  - Loss function L
  - L(ODEsolve(z(t_0), f, t_0, t_1, \omega))
    - To optimize L we need to know gradient with respect to the parameters
    - How does the gradient depend on z(t) at each time step ?
      - This is called adjoint, and it's dynamics is given by another equation (4) 
        - a(t)
        - This can also be solved by sending it to another ODE solver
    - Computing the gradients with respect to the parameters requirers a final equation 
      - integral{t_1}^{t_0} (Eq 5) and depends on both z(t) and a(t)
      - 
    - All the equations can be solved with a single call to the ode solver as a single vector
      - Described in algorithm 1
  - 
  - TODO : try to implement this

----
    
Limitations
- Minibatching is less straightforward
- Parameter for error rate
- 

----


Reading some other sources
- https://towardsdatascience.com/neural-odes-breakdown-of-another-deep-learning-breakthrough-3e78c7213795
  - More clear on the fact that methods like ResNet actually are ode's 
    - Did not think of this when I first read the paper, it's mentioned in the paper, but not in bold hehe
- https://nbviewer.org/github/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb
  - TODO go through this
- https://www.cs.toronto.edu/~rtqichen/pdfs/neural_ode_slides.pdf
  - Slide from the authors

