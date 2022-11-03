#include <stdlib.h>

void print_array(float *ptr, int length){
    printf("[");
    for(int i = 0 ; i < length; i++ ){
        if (i > 0) {
            printf(", ");
        }
        printf("%f", ptr[i]);
    }
    printf("]");
}
