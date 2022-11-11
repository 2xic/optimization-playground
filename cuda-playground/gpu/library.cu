#define IS_CUDA 1 

#include <stdlib.h>
#include "stdio.h"
#include "kernel.h"

extern "C" int testHello() {
    printf("hello python (from gpu!)!\n");
    return 0;
}

int init(){
    return 0;
}

