#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"

int main(){
    /*
        Need methods 
        -> CreateMatrix
            -> Create a matrix on the Gpu.
        -> SendToDevice
            -> Sends a matrix to device
                -> Replaces the data pointer with a device spesific one
                -> Need to be possible to send it back to the host
            -> Matrix has flag to know which device it's on
                -> GPU
                -> HOST
            -> 
        -> 
    */
    Matrix *a = createMatrixGpu(2, 2);
    setElement<<<1, 1>>>(a->data, a->columns, 0, 0, -2);
    setElement<<<1, 1>>>(a->data, a->columns, 0, 1, 1);
    setElement<<<1, 1>>>(a->data, a->columns, 1, 0, 0);
    setElement<<<1, 1>>>(a->data, a->columns, 1, 1, 4);

    Matrix *b = createMatrixGpu(2, 2);
    setElement<<<1,1>>>(b->data, b->columns, 0, 0, 6);
    setElement<<<1,1>>>(b->data, b->columns, 0, 1, 5);
    setElement<<<1,1>>>(b->data, b->columns, 1, 0, -7);
    setElement<<<1,1>>>(b->data, b->columns, 1, 1, 1);

    sendToHost(b);
    sendToGpu(b);

    // Results
    Matrix *c = GpuMatrixMatMul(a, b);

    c = GpuAdd(c, c);
    c = GpuAddConstant(c, 4.0, 0);

    sendToHost(c);

    float *c_host = c->data;
    printf("%f\n", c_host[0]);
    printf("%f\n", c_host[1]);
    printf("%f\n", c_host[2]);
    printf("%f\n", c_host[3]);
}
