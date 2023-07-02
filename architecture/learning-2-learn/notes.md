https://arxiv.org/pdf/1606.04474.pdf


# The idea
Idea is to use gradient decent to learn gradient decent.

Uses LSTM as memory to achieving this.

Idea is simple

x_{t + 1} = x_t + model(    f(x_t) )
model learns the output step size.

See equations on page 3 in the paper
