#include "../../../cpu/shared.h"

#ifndef gpu_library_h__
#define gpu_library_h__
 
extern int gpu_test(void);

extern int cpu_test(void);

extern void sendToHost(Matrix *m);
extern void sendToGpu(Matrix *m);

extern Matrix* GpuMatrixMatMul(Matrix*a, Matrix *b);

extern Matrix* GpuAdd(Matrix*a, Matrix *b);
extern Matrix* GpuAddConstant(Matrix *a, float b, int direction);

extern Matrix* GpuMul(Matrix *a, Matrix *b);
extern Matrix* GpuMulConstant(Matrix *a, float b, int direction);

extern Matrix* GpuDivideConstant(Matrix * a, float b, int direction);

extern Matrix* GpuSubtract(Matrix *a, Matrix *b);
extern Matrix* GpuSubtractConstant(Matrix *a, float b, int direction);

extern Matrix* GpuTranspose(Matrix*a);
extern Matrix *GpuExp(Matrix *a);


#endif  // gpu_library_h__
