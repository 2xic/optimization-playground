#ifndef shared_H_   
#define shared_H_

typedef struct {
    int *data;
    int rows;
    int columns;
} Matrix;

__global__ void createMatrix(void ** device, int rows, int columns);

__global__ void setElement(Matrix *a, int row, int col, int value);

__global__ void getElement( Matrix *a, int row, int col);

__global__ void MatMul(Matrix *a, Matrix *b, Matrix *c);

void print_array(int *ptr, int length);

#endif
