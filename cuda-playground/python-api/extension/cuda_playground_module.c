#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "tensor/tensor.c"
#include "copy.c"

Matrix *makeMatrix(int *arr);
void parse_array_recursive(PyObject *args, int *size_array);
void copy_recursive(PyObject *list_or_data, int indexLoc, int *Index, int length, Matrix* m);
int sizeOfPointer(int *arr);

static PyObject *
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

    TensorObject *obj = (TensorObject*)PyObject_CallObject((PyObject *)&TensorType, NULL);
    obj->matrix = createMatrix(i, j);


    return (PyObject*)obj;
}

static PyObject *parse_array(PyObject *self, PyObject *args)
{
    Py_ssize_t size = PyTuple_Size(args);

    if (size != 1)
    {
        printf("I only want one array!\n");
        return NULL;
    }
    int *size_array = malloc(128 * sizeof(int));
    parse_array_recursive(PyTuple_GetItem(args, 0), size_array);

    TensorObject *obj = (TensorObject*)PyObject_CallObject((PyObject *)&TensorType, NULL);
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


    return (PyObject*)obj;
}


static PyObject *
torch_exp(PyObject *self, PyObject *args)
{
    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    int ok;
    TensorObject *o;

    ok = PyArg_ParseTuple(args, "O", &o);
    if (!ok)
    {
        printf("Unexpected input\n");
        return NULL;
    }

    TensorObject *obj = (TensorObject*)PyObject_CallObject((PyObject *)&TensorType, NULL);
    obj->matrix = Exp(o->matrix);

    return (PyObject*)obj;
}

static PyMethodDef CudaplaygroundMethods[] = {
    {"tensor", tensor, METH_VARARGS,
     "Execute a shell command."},
    {"pare_array", parse_array, METH_VARARGS,
     "Execute a shell command."},
    {"exp", torch_exp, METH_VARARGS,
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
