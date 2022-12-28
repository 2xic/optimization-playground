#include "shared.h"


float(*create_subtract_extension(Matrix *m, float *constant))(int i, int j)
{
    float inside(int i, int j) {
        if (m != NULL){
            return getElement(m, i, j);
        } else {
            return *constant;
        }
    }
    return inside;
}

float(*create_operator_extension(float a, float b, int operator))(void)
{
    float apply_operator(void) {
        if (operator == ADD){
            return a + b;
        } else if (operator == SUBTRACT){
            return a - b;
        } else if (operator == MUL) {
            return a * b;
        } else if (operator == DIV){
            return a / b;
        }
    }
    return apply_operator;
}
