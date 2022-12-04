#include <Python.h>

#ifndef tensor_h__
#define tensor_h__
 
typedef struct
{
    PyObject_HEAD int size;
    Matrix *matrix;
} TensorObject;

static PyObject *tensor_add(PyObject *self, PyObject *args);
static PyObject *tensor_mul(PyObject *self, PyObject *args);
static PyObject *tensor_divide(PyObject *self, PyObject *args);
static PyObject *tensor_subtract(PyObject *self, PyObject *args);
static PyObject *tensor_transpose(TensorObject *self);
static PyObject *tensor_matmul(PyObject *self, PyObject *args);
static PyObject *tensor_negative(PyObject *self);
static PyObject *tensor_isEqual(PyObject *a, PyObject *b);
static PyObject *tensor_exp_direct(TensorObject *self, PyObject *Py_UNUSED(ignored));

static PyObject *Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored));
static PyObject *Ones(TensorObject *self, PyObject *Py_UNUSED(ignored));
static PyObject *Rand(TensorObject *self, PyObject *Py_UNUSED(ignored));
static PyObject *Print(TensorObject *self, PyObject *Py_UNUSED(ignored));

static PyObject *tensor_cuda(TensorObject *self);
static PyObject *tensor_host(TensorObject *self);

int getDevice(Matrix *a, Matrix *b);
void getTensorPointer(PyObject *a, PyObject *b, PyObject **tensor, PyObject **args, int *direction);

void tp_free(void *self);

#endif  // tensor_h__
