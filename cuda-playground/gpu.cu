#include <stdlib.h>
#include "stdio.h"
#include "shared.c"

#define N 1000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < N; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
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

    printf("Hello World from GPU!\n");
    print_array(out, N);
    
    free(a);
    cudaFree(d_a);
    free(b);
    cudaFree(d_b);    
    free(out);
    cudaFree(d_out);
}
