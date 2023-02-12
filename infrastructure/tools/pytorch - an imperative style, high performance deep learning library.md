[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/pdf/1912.01703.pdf)

- Other deep learning frameworks uses a static dataflow graph
  - Note that the paper was written in 2019, and since has for instance tensorflow also gotten eager exudation.
  - https://www.tensorflow.org/guide/effective_tf2

- PyTorch authors argues for dynamic eager execution since it helps with debugging
    - Eager execution means the result is returned immediately instead of it first constructing a computation graph that later has to be executed

    - Benefits of eager
      - More intuitive interface
      - Natural control flow
        - Python control flow instead of a control flow determined by graph
        - Make it easier to debug -> 

Design principles
- Be Pythonic
  - Components are easy to use, and interchangeable
- Researcher first
- Pragmatic performance
  - Performance, but not at the expense of simplicity and ease of use
  - Core is written in C++ (libtorch)
  - PyTorch is able to execute operators asynchronously and CPU-GPU synchronization is provided by the library, but can also be override
  - torch.multiprocessing to combat the Python global interpreter lock
    - i.e python being one thread
  -  Reference counting use to keep track of used tensors
     -  Freed once the reference counting reaches zero
  -  

