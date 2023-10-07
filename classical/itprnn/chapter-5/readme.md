## Symbol codes
Previously we have looked at block codes with a fixed length and in the previous chapter it was also loosy meaning that we lost information.
In this chapter we will look at symbol codes which are not lossy and are also variable in size. Most common codewords will have a shorter length and less common longer code words.

### What are (binary) symbol codes ? 
Ensemble of symbols codes into codes with the a string of values of 0 or 1
e.g.
- a -> 10000
- b -> 01000
- c -> 00100
- d -> 00001

One thing to note is that this mapping has to be unique so that we can correctly decode the code back to the symbol.
The best codes are the ones that we can easily know when are terminating (no codeword is prefix of another code). Symbol codes that have this property are called prefix codes.


## Kraft inequality
The code word length must satisfy the following equation to be uniquely decodable
$$\sum{i=1}^{I}2^{-l_i} <= 1$$ 

Codes that satisfy this are called complete codes.

## Huffman encoding
- We all know this algorithm from algorithm and data-structures
- This is also a optimal algorithm for symbol codes
- It does not handle the change in probability distribution very well and requires recomputation of the huffman codes
- We also coded it up [here](../../compression/huffman.py)

