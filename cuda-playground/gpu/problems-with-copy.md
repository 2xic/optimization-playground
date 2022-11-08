Looks like the probelm is that I called cudaMalloc on the device, but it should be called from the host.
```c
#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"

__global__ void setMatrix(int *d_a) {
    printf("malloc !\n");
    *d_a = 2;
}

int main(){
    int *a = (int*)malloc(10 * sizeof(int));
    int *d_a;
    cudaMalloc(&d_a, 2 * sizeof(int));

    *a = 4;

//    cudaMalloc((void**)&d_a, sizeof(float) * N);
    setMatrix<<<1, 1>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(int) * 2, cudaMemcpyDeviceToHost);

    printf("%i\n", a[0]);
}#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"

__global__ void setMatrix(int *d_a) {
    printf("malloc !\n");
    d_a[0] = 2;
    d_a[1] = 4;
//*(d_a+sizeof(int)) = 4;
}

int main(){
    int *a = (int*)malloc(10 * sizeof(int));
    int *d_a;
    cudaMalloc(&d_a, 10 * sizeof(int));

    a[0] = -1;

//    cudaMalloc((void**)&d_a, sizeof(float) * N);
    setMatrix<<<1, 1>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(int) * 4, cudaMemcpyDeviceToHost);

    printf("%i\n", a[0]);
    printf("%i\n", a[1]);
    printf("%i\n", a[2]);
}
```

Okay, so the problem is the way we are dealing with the pointer in the struct.

We should not dereference a struct pointer from the device on the host.

So, how do we deal with this ?
- Have a temp variable that deals with this :)
  - There was some discussion here.
  - https://stackoverflow.com/a/9323898

```c
#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"

struct Test {
    int *a;
    int isCuda;
};

__global__ void setMatrix(int *d_a) {
    printf("malloc !\n");
    d_a[0] = 2;
    d_a[1] = 4;
//*(d_a+sizeof(int)) = 4;
}

int main(){
    Test *a = (Test*)malloc(10 * sizeof(Test));
    int *arr;
    int *d_arr;
    cudaMalloc(&d_arr, 10 * sizeof(int));
    a->isCuda = true;
    a->a = d_arr;
    setMatrix<<<1, 1>>>(a->a);

    // alla do other stuff +++
    arr = (int*)malloc(10 * sizeof(int));
    cudaMemcpy(arr, a->a, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%i\n", arr[0]);
    printf("%i\n", arr[1]);
}
```
This works ^

