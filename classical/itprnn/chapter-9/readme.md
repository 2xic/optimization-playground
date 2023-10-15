## Communication over a noisy channel

Suppose we transmit 1000 bits per second with p0=p1=1/2 over a noisy channel that will flip a bit with probability of 10%. What is the rate of transmission of information ? 
- Guessing 900 bits (100% - error rate * information rate) might sounds good, but it is not the right answer as the receiver don't know when the noise happens.
- What might make more sense is the mutual information between the source and received signal.


We will now see if it is possible to communicate reliability over such a noisy channel.

## Review of the probability and information
*todo I should really just implement something for this*

## Noisy channels
`Q` describes the transition function from one input alphabet $X$  to an output alphabet $Y$

$$Q_{j|i} = P(y=b_j | x = a_i)$$

- Binary symmetric channels
- Binary erasure channel
- Noisy typewriter
- Z channel

## Inferring the input given the output
- One way is to use Bayes theorem on the output symbol and then predict what the input symbol might have been


## Information conveyed by a channel
The capacity of a channel $Q$ is 
$$
C(Q) = max_{P(X)}(I(X;Y))
$$

Where $P(X)$ is the optimal input distribution. 

## The noisy-channel coding theorem
Using long enough blockcodes you reduce the error rate.


### Resources
- [Noisy-channel coding theorem](https://en.wikipedia.org/wiki/Noisy-channel_coding_theorem)