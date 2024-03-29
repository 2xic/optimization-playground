#include <Python.h>
#include "../../cpu/shared.h"

void copy_recursive(PyObject *list_or_data, int indexLoc, int *Index, int length, Matrix *m)
{
    if (PyList_Check(list_or_data) != 1)
    {
        return;
    }
    int size = PyList_Size(list_or_data);
    for (int i = 0; i < size; i++)
    {
        *(Index + indexLoc) = i;

        PyObject *temp_p = PyList_GetItem(list_or_data, i);
        if (temp_p == NULL)
        {
            return;
        }

        if (PyList_Check(temp_p) != 1)
        {
            float value = PyFloat_AsDouble(temp_p);
            setElementN(m, Index, length, value);
        }
        else
        {
            copy_recursive(temp_p, indexLoc + 1, Index, length, m);
        }
    }
}

int sizeOfPointer(int *arr)
{
    int *copyArr = arr;
    int length = 0;

    while (*(copyArr + length) != '\0')
    {
        length++;
    }
    return length;
}

Matrix *makeMatrix(int *arr)
{
    int length = sizeOfPointer(arr);
    int stackArr[length + 1];
    for (int i = 0; i < length; i++)
    {
        stackArr[i] = (*arr);
        arr++;
    }

    return createMatrixN(stackArr, length);
}

void parse_array_recursive(PyObject *args, int *size_array)
{

    if (PyList_Check(args) != 1)
    {
        return;
    }
    Py_ssize_t size = PyList_Size(args);
    *size_array++ = (int)size;
    *(size_array + 1) = '\0';

    for (int i = 0; i < size; i++)
    {
        PyObject *temp_p = PyList_GetItem(args, i);
        if (temp_p == NULL)
        {
            return;
        }

        int is_not_list = PyList_Check(temp_p) != 1;

        if (is_not_list)
        {
            return;
        }
        else
        {
            parse_array_recursive(temp_p, size_array);
            return;
        }
    }
    return;
}
