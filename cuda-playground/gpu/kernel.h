#ifndef shared_H_   
#define shared_H_

typedef struct
{
    float *data;
    int rows;
    int columns;
    int *size;
    int device;
// TODO This is more like a tensor
} Matrix;

__global__ void createMatrix(void ** device, int rows, int columns);
__global__ void setElement(float *a, int columns, int row, int col, float value);
__global__ void getElement( Matrix *a, int row, int col);

extern "C" Matrix* MatrixMatMul(Matrix*a, Matrix *b);
Matrix *createMatrixGpu(int rows, int columns);

void print_array(int *ptr, int length);
extern "C" void sendToHost(Matrix *m);
extern "C" void sendToGpu(Matrix *m);

#endif
