[Neural Turing machine](https://arxiv.org/pdf/1410.5401.pdf)

- Analogous to a Turing Machine or Von Neumann architecture
- Architecture
  - External I/O goes into an and out from an controller
  - Controller does read and write from memory

  - All you need is a neural network controller and memory bank
  - 

## Opcodes
- Reading
  - M is memory bank with size N x M
  - Weights are normalized and each row sums to 1
  - Reading is a combination of row vectors from the memory and the read vector

- Writing
  - Controllers emits
    - a weighting, and a erase vector
    - a add vector which is added onto the memory

- Addressing mechanisms
  - How the weight vector is generated
    - Focusing by content
      - Equation 5
    - Focusing by location
      - Equation 7
  - 