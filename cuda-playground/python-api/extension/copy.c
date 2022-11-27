#include <Python.h>
#include "../../cpu/shared.h"

void copy_recursive(PyObject *list_or_data, int indexLoc, int *Index, int length, Matrix* m) {
    if (PyList_Check(list_or_data) != 1)
    {
        printf("NEVER CALLED\n");
        return;
    }
    int size = PyList_Size(list_or_data);
    if (indexLoc == 0) {
        printf("lengthhhh %i\n", size);
    }

    for (int i = 0; i < size; i++)
    {
        *(Index+indexLoc) = i;

        PyObject *temp_p = PyList_GetItem(list_or_data, i);
        if (temp_p == NULL)
        {
            printf("Quick skip :)\n");
            return;
        }

        if (indexLoc == 0) {
            printf("loc == %i\n", i);
        }

        /* Check if temp_p is numeric */
        if (PyList_Check(temp_p) != 1)
        {
            float value = PyFloat_AsDouble(temp_p);
            printf("%f\n", value);
            setElementN(m, Index, length, value);
        }
        else
        {
            copy_recursive(temp_p, indexLoc + 1, Index, length, m);
        }
    }
}


int sizeOfPointer(int *arr){
    int *copyArr = arr;
    int length = 0;
    
    while (*copyArr != '\0')
    {
        length++;
        *copyArr++;
    } 
    return length;
}

Matrix *makeMatrix(int *arr)
{
    printf("%p\n", arr);
    int length = sizeOfPointer(arr);
    printf("%p\n", arr);
    printf("%i\n", length);

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

        /* Check if temp_p is numeric */
        if (PyList_Check(temp_p) != 1)
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
