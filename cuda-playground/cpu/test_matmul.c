#include <cheat.h>
#include "shared.h"

CHEAT_TEST(mathematics_still_work,

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
