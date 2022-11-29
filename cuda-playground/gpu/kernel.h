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

extern "C" Matrix* MatrixMatMul(Matrix*a, Matrix *b);

extern "C" Matrix* MatrixAdd(Matrix*a, Matrix *b);
extern "C" Matrix *AddConstant(Matrix *a, float b, int direction);

extern "C" Matrix *Mul(Matrix *a, Matrix *b);
extern "C" Matrix *MulConstant(Matrix *a, float b, int direction);

extern "C" Matrix *DivideConstant(Matrix * a, float b, int direction);

extern "C" Matrix *Subtract(Matrix *a, Matrix *b);
extern "C" Matrix *SubtractConstant(Matrix *a, float b, int direction);

Matrix *createMatrixGpu(int rows, int columns);

void print_array(int *ptr, int length);
extern "C" void sendToHost(Matrix *m);
extern "C" void sendToGpu(Matrix *m);

#endif
