# Token Turing machines
[https://arxiv.org/pdf/2211.09119.pdf](https://arxiv.org/pdf/2211.09119.pdf)

- Autoregressive transformer model with memory
  - Inspired by Neural Turing machines
    - https://en.wikipedia.org/wiki/Neural_Turing_machine
      - RNN model of Turing machines
      - WIll 
    - Uses an external memory consisting of tokens that summaries the previous history
    - Transformer is used a s the processing unit

## Token turing machines
- The memory at time step t, consist of a set of m tokens of dimensionality d.
- The interface between the processing unit and memory are done with read and write operators
  - "Read" results is fed to the processing unit
  - "Write" is the output from the processing unit, and is then stored in the memory

- Memory should do selecting reading and writing
  - The model should therefore try to compress the content
  - I.e videos have much of the same frames, not everything is moving
  - Summarization is described in 2.1.1, but basically just learns a importance weight

- Reading from memory
  - Tokens in memory are concatenated with the input stream (I)
  - They also have a learned positional embedding to maker it possible for the model to distinguish tokens from memory and from the inputs

- Processing unit
  - Outputs a set of outputs tokens which are written back into memory
  - In addition to output tokens if necessary for the given tasks

- Writing to memory
  - Memory is erased if not re-selected based on the concat of current memory, input, and output tokens
  - 



