#ifndef shared_H_   
#define shared_H_

enum OPERATOR {
    SUBTRACT,
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

extern "C" Matrix* MatrixMatMul(Matrix*a, Matrix *b);

extern "C" Matrix* MatrixAdd(Matrix*a, Matrix *b);
extern "C" Matrix *AddConstant(Matrix *a, float b);

Matrix *createMatrixGpu(int rows, int columns);

void print_array(int *ptr, int length);
extern "C" void sendToHost(Matrix *m);
extern "C" void sendToGpu(Matrix *m);

#endif
