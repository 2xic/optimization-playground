
Wanted to take a quick look at cuda kernels
- https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/


Another thing I want to look at is SIMD / AMX instructions
- http://ftp.cvut.cz/kernel/people/geoff/cell/ps3-linux-docs/CellProgrammingTutorial/BasicsOfSIMDProgramming.html
- https://en.wikichip.org/wiki/x86/amx
- https://www.officedaytime.com/simd512e/

Thinking it might make sense to create a small cuda optimization library just to get a feeling for it. Basically just a Tensor class with cuda capabilities. 
To be extra fancy, we might also consider doing it with SIMD.

Just taking a quick look at this blogpost it seems actually quite straightforward. After all it's just pointers in memory. 
- https://leimao.github.io/blog/CUDA-Matrix-Multiplication/


Docs
- https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html
