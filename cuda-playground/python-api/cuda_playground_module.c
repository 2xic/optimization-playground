#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "external_library.h"
#include "tensor.c"

Matrix *makeMatrix(int *arr);
void parse_array_recursive(PyObject *args, int *size_array);
void copy_recursive(PyObject *list_or_data, int indexLoc, int *Index, int length, Matrix* m);
int sizeOfPointer(int *arr);

static PyObject *
gpu(PyObject *self, PyObject *args)
{
    gpu_test();

    return PyLong_FromLong(1);
}

static PyObject *
cpu(PyObject *self, PyObject *args)
{
    cpu_test();
    return PyLong_FromLong(1);
}


static TensorObject *
tensor(PyObject *self, PyObject *args)
{
    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }
    int ok;
    int i;
    int j;

    ok = PyArg_ParseTuple(args, "(ii)", &i, &j);
    if (!ok)
    {
        printf("Unexpected input\n");
        return NULL;
    }

    printf("Size (%i, %i)\n", i, j);

    TensorObject *obj = PyObject_CallObject((PyObject *)&TensorType, NULL);
    obj->matrix = createMatrix(i, j);


    return obj;
}


static TensorObject *parse_array(PyObject *self, PyObject *args)
{
    Py_ssize_t size = PyTuple_Size(args);

    if (size != 1)
    {
        printf("I only want one array!\n");
        return NULL;
    }
    int *size_array = malloc(128 * sizeof(int));
    parse_array_recursive(PyTuple_GetItem(args, 0), size_array);

    TensorObject *obj = PyObject_CallObject((PyObject *)&TensorType, NULL);
    Matrix *m = makeMatrix(size_array);
    obj->matrix = m;


    int *indexArray = malloc(sizeOfPointer(size_array) * sizeof(int*));
    memset(indexArray, 0, sizeOfPointer(size_array) * sizeof(int*));
    printf("arr_size = %i\n", sizeOfPointer(size_array));
  //  printf("%i %i\n", *indexArray, *(indexArray + 1));
    copy_recursive(
        PyTuple_GetItem(args, 0), 
        0,
        indexArray,
        sizeOfPointer(size_array),
        obj->matrix
    );
    // setElementN(b, (int[]){1, 0}, -7);


    return obj;
}


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
            /*            */
            /*
            printf("Write number :)\n");
            for(int i = 0; i < length; i++){
                if(i > 0) {
                    printf(",");
                }
                printf("%i ", *(Index + i));
            }
            printf("\n");
            */
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

static PyMethodDef CudaplaygroundMethods[] = {
    {"gpu", gpu, METH_VARARGS,
     "Execute a shell command."},
    {"cpu", cpu, METH_VARARGS,
     "Execute a shell command."},
    {"tensor", tensor, METH_VARARGS,
     "Execute a shell command."},
    {"pare_array", parse_array, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef Cudaplaygroundmodule = {
    PyModuleDef_HEAD_INIT,
    "cudaplayground",
    NULL,
    -1,
    CudaplaygroundMethods};

PyMODINIT_FUNC
PyInit_cudaplayground(void)
{
    PyObject *m;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;

    m = PyModule_Create(&Cudaplaygroundmodule);
    if (m == NULL)
        return NULL;

    return m;
}
