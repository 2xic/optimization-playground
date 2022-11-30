#ifndef kernel_H_   
#define kernel_H_

enum OPERATOR {
    SUB,
    ADD,
    MUL,
    DIV
};


typedef struct
{
    float *data;
    int rows;
    int columns;
    int *size;
    int device;
// TODO This is more like a tensor
} Matrix;

__global__ void MatMul(float *a, float *b, float *c, int columns, int rows);
__global__ void createMatrix(void ** device, int rows, int columns);
__global__ void setElement(float *a, int columns, int row, int col, float value);

extern "C" Matrix* GpuMatrixMatMul(Matrix*a, Matrix *b);

extern "C" Matrix* GpuAdd(Matrix*a, Matrix *b);
extern "C" Matrix* GpuAddConstant(Matrix *a, float b, int direction);

extern "C" Matrix* GpuMul(Matrix *a, Matrix *b);
extern "C" Matrix* GpuMulConstant(Matrix *a, float b, int direction);

extern "C" Matrix* GpuDivideConstant(Matrix * a, float b, int direction);

extern "C" Matrix* GpuSubtract(Matrix *a, Matrix *b);
extern "C" Matrix* GpuSubtractConstant(Matrix *a, float b, int direction);

extern "C" Matrix* GpuTranspose(Matrix*a);

extern "C" Matrix *GpuExp(Matrix *a);

Matrix *createMatrixGpu(int rows, int columns);

void print_array(int *ptr, int length);
extern "C" void sendToHost(Matrix *m);
extern "C" void sendToGpu(Matrix *m);

#endif
