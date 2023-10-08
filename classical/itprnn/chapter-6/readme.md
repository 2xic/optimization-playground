## stream codes
In this chapter we will discuss two data compression schemes
- Arithmetic coding (which we already wrote a tiny implementation for in [../../compression/arithmetic_coding.py](../../compression/arithmetic_coding.py)). Many state of the art compression algorithms uses this method.
- Lempel-Ziv coding which is a universal method that does a reasonable job for any source. 

## Arithmetic codes
The encoder creates a binary string based on the probability of symbols. There will be used together what are called intervals i.e [0.01, 0.10)].
What are intervals ? They are mappings to the alphabet symbols. Longer binary strings corresponds to a smaller interval, and shorter strings to larger intervals.

### Encoding
The pseudo code attached with the example in the book is the following 
```
u = 0
u = 1
p = v - u

// N = size of string
for n = 1 to N {
    v = u + p * (R_n (X_n))
    u = u + p * (Q_n (X_n))
    p = v - u
}
```
The process of creating the intervals can be defined in terms of the lower and upper cumulative probabilities (which is what Q and R defines).

There is also a example of this process on page 126 in the [pdf version](https://www.inference.org.uk/itprnn/book.pdf). 

### Decoding
I find the way things are described here a bit hard to follow - I liked this [document](https://www.cs.cmu.edu/~aarti/Class/10704/Intro_Arith_coding.pdf) more and this was also used for writing a toy implementation of [arithmetic_coding.py](../../compression/arithmetic_coding.py).

## Lempel-Ziv coding
(used by gzip - `LZ77')
THe method is to replace a substring with a pointer to an earlier occurrence of that substring. 

You can see an example (basic) implementation [here](../../compression/lwz.py)
