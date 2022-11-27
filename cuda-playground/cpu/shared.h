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

#ifndef IS_CUDA
    void print_array(float *ptr, int length);
#else
    extern "C" void print_array(float *ptr, int length);
#endif

int getSize(int size[]);

float getElement(Matrix *a, int row, int col);

void setElement(Matrix *a, int row, int col, float value);

void freeMatrix(Matrix *a);

Matrix *createMatrix(int rows, int columns);

void fillRandom(Matrix *a);

Matrix *createMatrixN(int size[], int length);

Matrix *Add(Matrix *a, Matrix *b);
Matrix *AddConstant(Matrix *a, float b, int direction);

Matrix *Mul(Matrix *a, Matrix *b);
Matrix *MulConstant(Matrix *a, float b, int direction);

Matrix *DivideConstant(Matrix * a, float b, int direction);

Matrix *Subtract(Matrix *a, Matrix *b);
Matrix *SubtractConstant(Matrix *a, float b, int direction);

int isEqual(Matrix *a, Matrix *b);

Matrix *Exp(Matrix * a);

void fill(Matrix *a, int value);

void setElementN(Matrix *a, int *location, int length, float value);

#ifndef IS_CUDA
    Matrix *MatMul(Matrix *a, Matrix *b);
#else
    extern "C" __global__ Matrix *MatMul(Matrix *a, Matrix *b);
#endif

#endif
