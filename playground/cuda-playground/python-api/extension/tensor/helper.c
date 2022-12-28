#include "external_library.h"
#include <Python.h>

// One of the objects will always be a tensor object
void getTensorPointer(PyObject *a, PyObject *b, PyObject **tensor, PyObject **args, int *direction)
{
    if (PyLong_Check(a))
    {
        *tensor = b;
        *args = a;
        *direction = 1;
    }
    else
    {
        *tensor = a;
        *args = b;
        *direction = 0;
    }
}

int getDevice(Matrix *a, Matrix *b)
{
    if (a != NULL && b != NULL && a->device == b->device)
    {
        return a->device;
    }
    else if (a != NULL && b == NULL)
    {
        return a->device;
    }
    else
    {
        return -1;
    }
}
