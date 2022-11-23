#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "external_library.h"
#include "tensor.c"

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
    if (!ok) {
        printf("Unexpected input\n");
        return NULL;
    }

    printf("Size (%i, %i)\n", i, j);

    TensorObject *obj = PyObject_CallObject((PyObject *)&TensorType, NULL);    
    obj->matrix = createMatrix(i, j);

    return obj;
}



static PyMethodDef CudaplaygroundMethods[] = {
    {"gpu", gpu, METH_VARARGS,
     "Execute a shell command."},
    {"cpu", cpu, METH_VARARGS,
     "Execute a shell command."},
    {"tensor", tensor, METH_VARARGS,
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





