#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"


int main(){
    printf("Creating first matrix :)\n");

	Matrix *d_a;
    createMatrix<<<1, 1>>>((void**)&d_a, 2, 2);
    setElement<<<1, 1>>>(d_a, 0, 0, -2);
    setElement<<<1, 1>>>(d_a, 0, 1, 1);
    setElement<<<1, 1>>>(d_a, 1, 0, 0);
    setElement<<<1, 1>>>(d_a, 1, 1, 4);

    printf("Creating second matrix :)\n");

	Matrix *d_b;
    createMatrix<<<1, 1>>>((void**)&d_b, 2, 2);
    setElement<<<1, 1>>>(d_b, 0, 0, -2);
    setElement<<<1, 1>>>(d_b, 0, 1, 1);
    setElement<<<1, 1>>>(d_b, 1, 0, 0);
    setElement<<<1, 1>>>(d_b, 1, 1, 4);

    printf("Creating last matrix :)\n");

    Matrix *d_c;
    createMatrix<<<2, 1>>>((void**)&d_c, 2, 2);
    cudaMemset(d_c->data, 0, 2 * 2);

    MatMul<<<2, 1>>>(d_a, d_b, d_c);
    printf("Finished  matmul :D\n");


    int *out;
    printf("copy ?? ");
    out = (int*) malloc(sizeof(int *) * 2 * 2);
    cudaMemcpy(out, d_c->data, sizeof(int) * 2 * 2, cudaMemcpyDeviceToHost);
    printf("printing ?? ");
    print_array(out, 2);

    printf("done \n");

    cudaFree(d_a->data);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b->data);
    cudaFree(d_c);
    cudaFree(d_c->data);

///    MatMul<<<1, 1>>>(d_a, d_b, d_c);

    /*
    Matrix *results;
    createMatrix<<<1,1>>>(
        a->rows,
        b->columns
    );
    */


    /*
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    out = (float*)malloc(sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);
   // cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);

    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    free(a);
    cudaFree(d_a);
    free(b);
    cudaFree(d_b);    
    free(out);
    cudaFree(d_out);
    */
}
