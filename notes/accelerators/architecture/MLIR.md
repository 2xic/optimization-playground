## [MLIR](https://arxiv.org/pdf/2002.11054.pdf)

### Introduction
- Aims to address challenges with domain specific compilers
- Many of the existing solution has characteristic of being on a single abstraction level   
  - LLVM IR -> "C with vectors"
  - JVM -> Object oriented type system with garbage collection
- Some problems require high level and some low level abstractions
  - Code analysis of C++ code is hard on just IR
- Languages / Machine learning frameworks develop custom solutions use custom IR to solve their domain problems
  - MLIR wants to simply this

## How does MLIR simplify things ? 
- SSA instructions are standardized IR-Based data structures
- Declare system for defining IR dialects
- Proving core infrastructure
  - Parsing 
  - Location tracking
  - multithread compilation support

## Where did this all originate from ?  
Tensorflow graph needed to support core ml, tensorflow lite, XLA, etc this then had it's own set of custom IRs (LLVM / TPU IR / +++).

## The problem
Many of the issues that software like LLVM tries to solve is recreated because of the new abstractions on top.

## The solution
Flexible IR 

