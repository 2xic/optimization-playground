## [triton](https://openai.com/research/triton)
- [Source code](https://github.com/openai/triton)
- [Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

Other libraries like [Numba](https://numba.pydata.org/) uses [SIMT execution](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads), but in triton the idea of thread index is removed. Thread index is something used by Cuda and is something both my self and others have found annoying. All functions are executed in the same thread. Downside here is that we loose some parallel computation, but gain simplicity of the logic. 

General aritecture is like this
- Python code goes through an AST visitor and into a Triton-IR
- The Triton compiler uses this and feeds it into the LLVM-IR
- This then gets converted by libLLVM into [Parallel Thread Execution (PTX)](https://en.wikipedia.org/wiki/Parallel_Thread_Execution) that is executed on the machine.

[Chat with Philippe Tillet, Author of Triton](https://www.mlsys.ai/papers/tillet.html)
^ good chat with one of the core authors

## Resources
- [Article on the journey into LLVM IR](https://un-devs.github.io/low-level-exploration/journey-to-understanding-llvm-ir/#)
- Oh actually [LLVM](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) has quite a good tutorial on all of this
- This article on the [LLVM IR](https://mukulrathi.com/create-your-own-programming-language/llvm-ir-cpp-api-tutorial/) is also great
- [Writing a LLVM backend](https://llvm.org/docs/WritingAnLLVMBackend.html)
- 
