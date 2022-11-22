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


static PyObject *
tensor_f(PyObject *self, PyObject *args)
{
    printf("Hello bro!\n");
    // Py_INCREF(&TensorType);
  if (PyType_Ready(&TensorType)) {
    return NULL;
  } 

    /* Pass two arguments, a string and an int. */
    PyObject *argList = Py_BuildValue("si", "hello", 42);

    /* Call the class object. */
    TensorObject *obj = PyObject_CallObject((PyObject *) &TensorType, argList);
    Py_INCREF(obj);
    printf("New matrix created ? %p \n ", obj);
    obj->matrix = createMatrix(4, 4);
    printf("yeah boiii\n");




    return obj;
   }



static PyMethodDef CudaplaygroundMethods[] = {
    {"gpu", gpu, METH_VARARGS,
     "Execute a shell command."},
    {"cpu", cpu, METH_VARARGS,
     "Execute a shell command."},
    {"tensor_f", tensor_f, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef Cudaplaygroundmodule = {
    PyModuleDef_HEAD_INIT,
    "cudaplayground",
    NULL,
    -1,
    CudaplaygroundMethods};


/*
int
PyModule_AddType(PyObject *module, const char *name, PyTypeObject *type)
{
    if (PyType_Ready(type)) {
        return -1;
    }
    Py_INCREF(type);
    if (PyModule_AddObject(module, name, (PyObject *)type)) {
        Py_DECREF(type);
        return -1;
    }
    return 0;
}
*/
PyMODINIT_FUNC
PyInit_cudaplayground(void)
{
    PyObject *m;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;

    m = PyModule_Create(&Cudaplaygroundmodule);
    if (m == NULL)
        return NULL;

  //  PyModule_AddType(m, (PyObject *)&TensorType);
    //PyModule_AddType(m, "tensor", (PyObject *)&TensorType);
    /*
    
    
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *) &TensorType) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(m);
        return NULL;
    }
    */
   
   

    return m;
}

