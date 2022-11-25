#include <cheat.h>
#include "shared.h"

CHEAT_TEST(mathematics_still_work,

	Matrix *a = createMatrix(2, 2);
    setElement(a, 0, 0, -2);
    setElement(a, 0, 1, 1);
    setElement(a, 1, 0, 0);
    setElement(a, 1, 1, 4);
    int s[] = {2, 2};
    Matrix *b = createMatrixN(s, 2);

    setElementN(b, (int[]){0, 0}, 2, 6);
    setElementN(b, (int[]){0, 1}, 2, 5);
    setElementN(b, (int[]){1, 0}, 2, -7);
    setElementN(b, (int[]){1, 1}, 2, 1);

    printf("%i\n", getElement(b, 0, 0));

    Matrix *c = MatMul(a, b);
    cheat_assert(getElement(c, 0,0) == -19);
    cheat_assert(getElement(c, 0,1) == -9);
    cheat_assert(getElement(c, 1,0) == -28);
    cheat_assert(getElement(c, 1,1) == 4);

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(c);

    cheat_assert(2 + 2 == 4);
	cheat_assert_not(2 + 2 == 5);
)


