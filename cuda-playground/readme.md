Just a tiny cuda playground.

Mainly I wanted to get a feeling for how the CUDA interface works. Ended up implementing the basic matrix operators needed for implementing a simple NN with a python interface for a cpu and gpu.

Please note that this a playground(!!!), and there are multiple things that could be done to improve it if you wanted to make it into something real. 
Here are a few things on top of my mind :
- The CPU and GPU interface implementation should be a shared header, and not partially implemented in two headers.
- Some variables like (Direction) should be a enums, not int
- Currently exception messages are not set in the python interface
- Naming convention use underscore (`_`) as word seperator, and not camelCase
- I don't think deallocation is done correctly in the Python extension currently, so that should be fixed.
  - I also don't think the reference count is done correctly, and these bugs are related.
- Tensor and matrix variable usage is bit ambiguous, should only have tensor object and have the matrix run if the size is == 2
  - Same is true with SIZE and length. PyTorch calls it shape, maybe shape size is best name.
