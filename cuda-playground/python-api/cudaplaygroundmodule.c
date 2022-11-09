#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gpu_library.h"

static PyObject *
cudaplayground_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef CudaplaygroundMethods[] = {
    {"system",  cudaplayground_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef Cudaplaygroundmodule = {
    PyModuleDef_HEAD_INIT,
    "cudaplayground",
    NULL,
    -1,
    CudaplaygroundMethods
};








PyMODINIT_FUNC
PyInit_cudaplayground(void)
{
    testHello();
    return PyModule_Create(&Cudaplaygroundmodule);
}

int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("cudaplayground", PyInit_cudaplayground) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject *pmodule = PyImport_ImportModule("cudaplayground");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'cudaplayground'\n");
    }

    PyMem_RawFree(program);
    return 0;
}
