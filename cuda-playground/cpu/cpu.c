#include <stdlib.h>
#include "stdio.h"
#include "shared.c"

void vector_add(float *out, float *a, float *b, int N) {
    for(int i = 0; i < N; i++){
        out[i] = a[i] + b[i];
    }
}

#define N 1000

int main(){
    float *a, *b, *out; 

    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    for(int i = 0; i < N; i++){
        a[i] = 1.0f; 
        b[i] = 2.0f;
    }

    vector_add(out, a, b, N);

    printf("Hello World from GPU!\n");
    print_array(out, N);
    
    free(a);
    free(b);
    free(out);

    return 0;
}
