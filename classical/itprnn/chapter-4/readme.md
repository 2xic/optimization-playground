# The Source Coding Theorem
In this chapter we will look at 
- Shannon information is a sensible way of measuring information
- Entropy of the ensemble is a sensible measure of the average information
  
The less probable a outcome is, the higher is the Shannon information content.

### Shannon information
- One nice things by it being logs is the additive feature of logs over multiplication.
- There is also the guessing game which I think was mentioned in the original paper by Shannon (how many yes / no question do you need to ask to know the answer)
  - "Improbable outcomes do convey more information than probable outcomes"

### Information content defined terms of lossy compression
We have two text files
1. All letters are used with equal probability
2. All letters are used with the same probability as they commonly are used in real world text
Can we defined a measure of information that distinguishes between theses two files ? 
- The first file is obviously more predictable and contains less information
- We can create reduced code to for document two to omit certain uncommonly used characters ({, [, ]}, !, ? etc). There more characters we omit, the less we need to store. 

By doing this reduction we can make the analogy that the essential bit content of the text is the log2 of the values in the set of the probability <= to the risk tolerance we selected (H_sigma = log_2 |S| ).

The point the author is aiming at is when the number of symbols are combined into an ensemble of size `N` and `N` becomes very large the results will be that `N* H_sigma(X^N )` will approach the entropy of `H(X)`

### Shannon’s Source Coding Theorem
The point here is somehow the source coding theorem, but I don't see the direct connection. Anyways [this](https://mbernste.github.io/posts/sourcecoding/) is a good article on the source coding theorem and states that the smallest possible expected word length will be the entropy of the ensemble.

## Long strings probability
The probability of a long string will be the product of all it's symbol probabilities and information content is the number of symbols plus the probability of the symbols times their log2(1/p) value

## Other resources
- [Shannon's source coding theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem)

