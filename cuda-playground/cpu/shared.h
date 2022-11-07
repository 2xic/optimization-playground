#ifndef shared_H_   
#define shared_H_


typedef struct {
    int *data;
    int rows;
    int columns;
} Matrix;

#ifndef IS_CUDA
    void print_array(float *ptr, int length);
#else
    extern "C" void print_array(float *ptr, int length);
#endif

int getElement(Matrix *a, int row, int col);

void setElement(Matrix *a, int row, int col, int value);

void freeMatrix(Matrix *a);

Matrix *createMatrix(int rows, int columns);


#ifndef IS_CUDA
    Matrix *MatMul(Matrix *a, Matrix *b);
#else
    extern "C" __global__ Matrix *MatMul(Matrix *a, Matrix *b);
#endif

#endif
