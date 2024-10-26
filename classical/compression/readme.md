https://en.wikipedia.org/wiki/Category:Data_compression

### Benchmarks
- [Hutter price](http://prize.hutter1.net/)
- [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html)

Currently [nncp](https://bellard.org/nncp/nncp.pdf) and variants of [cmix](https://github.com/kaitz/fx-cmix) are on top of both leaderboard.

### [Data Compression Explained](http://mattmahoney.net/dc/dce.html#Section_1)
All data compression algorithms are based on 
- a model that estimates the probability of distribution (E is more probably than X)
- a coder that assign shorter codes to more common words

There are no [universal compression](http://mattmahoney.net/dc/dce.html#Section_11) and the [codes](http://mattmahoney.net/dc/dce.html#Section_12) are also bounded. 

*"Compression = modeling + coding. Coding is a solved problem. Modeling is provably not solvable."*

The page has many code snippets for various compression algorithms.

### [SEQUENTIAL NEURAL TEXT COMPRESSION](https://people.idsia.ch/~juergen/ieeetnn1996.pdf)
- They combine neural networks and traditional methods (huffman coding and arithmetic coding) 
- Variants of predictive coding is used. Neural network is used to learn the conditional probability of the next character given the previous one.
- They train off-line and on-line variants
  - off-line = network trained ones and frozen. The network is part of the compression code and is frozen when used 
  - on-line = More expensive, but could preform better by using insights from the specific file being compressed
- Method 1
  - The networks creates a probability of characters and feed it into huffman

### [PAQ](http://mattmahoney.net/dc/dce.html#Section_436)
- Builds upon work from [Schmidhuber and Heil (1996)](https://people.idsia.ch/~juergen/ieeetnn1996.pdf)
- Improvements over the original work includes
  - Predict a bit over time instead of a character
  - Make the training online
  - Use a hash function to select a neuron instead of a layer to reduce the activity of the network
- 

### [cmix](http://www.byronknoll.com/cmix.html)
- [introduction blogpost](http://byronknoll.blogspot.com/2014/01/cmix.html) of Cmix also refers the data compression book.
- It is build up by three main components
  - Preprocessing
  - Model prediction
  - Context mixing
- Preprocessing transforms the input data into a way that is more easily to compress. The data is compressed with a single pass with one bit at the time. This probability is encoded as arithmetic coding.
- Independent models are used to predict the next probability of each input stream. 


[fx-cmix](https://github.com/kaitz/fx-cmix/tree/main/src) is a modification of this and won the latest (2024) hutter prize. There is some discussion on it [here](https://encode.su/threads/4161-fx-cmix-(HP)?p=81533&viewfull=1#post81533). Most of the code is on [kaitz](https://github.com/kaitz) github profile.

### [DeepZip: Lossless Compression using Recurrent Networks](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761006.pdf)
- Can RNNs be used for data compression ? 
- Components
  - RNN Probability Estimator block
  - Arithmetic Coder Block
- Haven't I seen this before ? 

## Other resources
-  [Zip Files: History, Explanation and Implementation](https://www.hanshq.net/zip.html) was a fun read.
-  [Data compression](https://en.wikipedia.org/wiki/Data_compression) on Wikipedia is quite alright


