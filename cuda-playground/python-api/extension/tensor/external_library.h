#include "../../../cpu/shared.h"

#ifndef gpu_library_h__
#define gpu_library_h__
 
extern int gpu_test(void);

extern int cpu_test(void);

extern void sendToHost(Matrix *m);
extern void sendToGpu(Matrix *m);

extern Matrix* MatrixMatMul(Matrix*a, Matrix *b);

#endif  // gpu_library_h__
