#include "shared.h"
#include <stdio.h>

int cpu_test()
{

    printf("hello python (from cpu!)!\n");

    Matrix *a = createMatrix(2, 2);
    setElement(a, 0, 0, -2);
    setElement(a, 0, 1, 1);
    setElement(a, 1, 0, 0);
    setElement(a, 1, 1, 4);
    Matrix *b = createMatrix(2, 2);

    setElement(b, 0, 0, 6);
    setElement(b, 0, 1, 5);
    setElement(b, 1, 0, -7);
    setElement(b, 1, 1, 1);

    Matrix *c = MatMul(a, b);
    printf("%f\n", getElement(c, 0, 0));
    printf("%f\n", getElement(c, 0, 1));
    printf("%f\n", getElement(c, 1, 0));
    printf("%f\n", getElement(c, 1, 1));

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(c);

    return 0;
}

