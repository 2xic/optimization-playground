#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"


int main(){
    int SIZE = 4;
    int *a_device;
    Matrix *a = (Matrix*)malloc(sizeof(Matrix));
    cudaMalloc(&a_device, SIZE * sizeof(int));
    a->data = a_device;
    a->rows = 2;
    a->columns = 2;

    setElement<<<1, 1>>>(a->data, a->columns, 0, 0, -2);
    setElement<<<1, 1>>>(a->data, a->columns, 0, 1, 1);
    setElement<<<1, 1>>>(a->data, a->columns, 1, 0, 0);
    setElement<<<1, 1>>>(a->data, a->columns, 1, 1, 4);

    int *b_device;
    Matrix *b = (Matrix*)malloc(sizeof(Matrix));
    cudaMalloc(&b_device, SIZE * sizeof(int));
    b->data = b_device;
    b->rows = 2;
    b->columns = 2;

    setElement<<<1,1>>>(b->data, b->columns, 0, 0, 6);
    setElement<<<1,1>>>(b->data, b->columns, 0, 1, 5);
    setElement<<<1,1>>>(b->data, b->columns, 1, 0, -7);
    setElement<<<1,1>>>(b->data, b->columns, 1, 1, 1);

    int *c_device;
    cudaMalloc(&c_device, SIZE * sizeof(int));
    MatMul<<<1, 1>>>(a->data, b->data, c_device, a->rows, b->columns);

    int *c_host;
    c_host = (int*)malloc(SIZE * sizeof(int));
    cudaMemcpy(c_host, c_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%i\n", c_host[0]);
    printf("%i\n", c_host[1]);
    printf("%i\n", c_host[2]);
    printf("%i\n", c_host[3]);
}
