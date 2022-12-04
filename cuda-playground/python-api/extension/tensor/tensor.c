// https://docs.python.org/3/extending/newtypes_tutorial.html

#include <Python.h>
#include "../../../cpu/shared.h"
#include "external_library.h"
#include "tensor.h"
#include "helper.c"

PyNumberMethods magic_num_methods = {
    .nb_add = tensor_add,
    .nb_negative = tensor_negative,
    .nb_true_divide = tensor_divide,
    .nb_multiply = tensor_mul,
    .nb_subtract = tensor_subtract};

static PyMethodDef tensor_methods[] = {
    {"zeros", (PyCFunction)Zeros, METH_NOARGS, "Creates a zero tensor"},
    {"ones", (PyCFunction)Ones, METH_NOARGS, "Creates a ones tensor"},
    {"rand", (PyCFunction)Rand, METH_NOARGS, "Creates a random tensor"},
    {"T", (PyCFunction)tensor_transpose, METH_NOARGS, "Transpose tensor (unoptimized)"},
    {"matmul", (PyCFunction)tensor_matmul, METH_O, "Matmul two tensors"},
    {"print", (PyCFunction)Print, METH_NOARGS, "Print the tensor"},
    {"isEqual", (PyCFunction)tensor_isEqual, METH_O, "Matmul two tensors"},
    {"cuda", (PyCFunction)tensor_cuda, METH_NOARGS, "Send tensor to cuda device"},
    {"host", (PyCFunction)tensor_host, METH_NOARGS, "Sends tensor to host"},
    {"exp", (PyCFunction)tensor_exp_direct, METH_NOARGS, "Sends tensor to host"},
    {NULL} /* Sentinel */
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "tensor",
    .tp_doc = PyDoc_STR("Tensor objects"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = sizeof(Matrix),
    .tp_new = PyType_GenericNew,
    .tp_free = tp_free,
    .tp_as_number = &magic_num_methods,
    .tp_methods = tensor_methods,
};

void tp_free(void *selfx)
{

    //   printf("free the beef? +\n");
}

static PyObject *
Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    return (PyObject *)self;
}

static PyObject *
Ones(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    fill(self->matrix, 1);
    self->size = 42;
    return (PyObject *)self;
}

static PyObject *
Rand(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    fillRandom(self->matrix);
    return (PyObject *)self;
}

static PyObject *
tensor_exp_direct(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
    if (self->matrix->device == 0)
    {
        obj->matrix = Exp(self->matrix);
    }
    else if (self->matrix->device == 1)
    {
        obj->matrix = GpuExp(self->matrix);
    }
    else
    {
        // Throw an error
        return NULL;
    }
    return (PyObject *)obj;
}

static PyObject *
Print(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("Matrix pointer = %p (%i, %i)\n", self, self->matrix->columns, self->matrix->rows);

    for (int i = 0; i < self->matrix->rows; i++)
    {
        for (int j = 0; j < self->matrix->columns; j++)
        {
            if (j > 0)
            {
                printf(", ");
            }
            printf("%f", getElement(self->matrix, i, j));
        }
        printf("\n");
    }

    return (PyObject *)self;
}

static PyObject *
tensor_subtract(PyObject *a, PyObject *b)
{
    PyObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction);

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *tensorSelf = (TensorObject *)self;

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensorSelf->matrix, NULL);

        if (device == 0)
        {
            obj->matrix = SubtractConstant(tensorSelf->matrix, value, direction);
        }
        else if (device == 1)
        {
            obj->matrix = GpuSubtractConstant(tensorSelf->matrix, value, direction);
        }

        return (PyObject *)obj;
    }
    else
    {

        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensorSelf->matrix, ((TensorObject *)args)->matrix);

        if (device == 0)
        {
            obj->matrix = Subtract(tensorSelf->matrix, ((TensorObject *)args)->matrix);
        }
        else if (device == 1)
        {
            obj->matrix = GpuSubtract(tensorSelf->matrix, ((TensorObject *)args)->matrix);
        }

        return (PyObject *)obj;
    }
}

static PyObject *
tensor_negative(PyObject *self)
{
    TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
    TensorObject *tensorSelf = ((TensorObject *)self);
    if (tensorSelf->matrix->device == 0)
    {
        obj->matrix = MulConstant(tensorSelf->matrix, -1, 0);
    }
    else if (tensorSelf->matrix->device == 1)
    {
        obj->matrix = GpuMulConstant(tensorSelf->matrix, -1, 0);
    }

    return (PyObject *)obj;
}

static PyObject *
tensor_add(PyObject *a, PyObject *b)
{
    PyObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction);

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *tensor_self = (TensorObject *)self;

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensor_self->matrix, NULL);

        if (device == 0)
        {
            obj->matrix = AddConstant(tensor_self->matrix, value, direction);
        }
        else if (device == 1)
        {
            obj->matrix = GpuAddConstant(tensor_self->matrix, value, direction);
        }

        return (PyObject *)obj;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensor_self->matrix, ((TensorObject *)args)->matrix);

        if (device == 0)
        {
            obj->matrix = Add(tensor_self->matrix, ((TensorObject *)args)->matrix);
        }
        else if (device == 1)
        {
            obj->matrix = GpuAdd(tensor_self->matrix, ((TensorObject *)args)->matrix);
        }

        return (PyObject *)obj;
    }
}

static PyObject *
tensor_mul(PyObject *a, PyObject *b)
{
    PyObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction);

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *tensor_self = (TensorObject *)self;

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensor_self->matrix, NULL);

        if (device == 0)
        {
            obj->matrix = MulConstant(tensor_self->matrix, value, direction);
        }
        else if (device == 1)
        {
            obj->matrix = GpuMulConstant(tensor_self->matrix, value, direction);
        }

        return (PyObject *)obj;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        int device = getDevice(tensor_self->matrix, ((TensorObject *)args)->matrix);

        if (device == 0)
        {
            obj->matrix = Mul(tensor_self->matrix, ((TensorObject *)args)->matrix);
        }
        else if (device == 1)
        {
            obj->matrix = GpuMul(tensor_self->matrix, ((TensorObject *)args)->matrix);
        }

        return (PyObject *)obj;
    }
}

static PyObject *
tensor_divide(PyObject *a, PyObject *b)
{
    PyObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction);

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *tensorSelf = (TensorObject *)self;

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);

        int device = getDevice((tensorSelf)->matrix, NULL);

        if (device == 0)
        {
            obj->matrix = DivideConstant((tensorSelf)->matrix, value, direction);
        }
        else if (device == 1)
        {
            obj->matrix = GpuDivideConstant((tensorSelf)->matrix, value, direction);
        }

        return (PyObject *)obj;
    }
    else
    {
        // TODO set error
        return NULL;
    }
}

static PyObject *
tensor_transpose(TensorObject *self)
{

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
    int s[] = {self->matrix->columns, self->matrix->rows};

    if (self->matrix->device == 0)
    {
        Matrix *results = createMatrixN(s, 2);
        obj->matrix = results;

        for (int i = 0; i < self->matrix->rows; i++)
        {
            for (int j = 0; j < self->matrix->columns; j++)
            {
                setElement(obj->matrix, j, i, getElement(self->matrix, i, j));
            }
        }
    }
    else if (self->matrix->device == 1)
    {
        obj->matrix = GpuTranspose(self->matrix);
    }

    return (PyObject *)obj;
}

static PyObject *
tensor_matmul(PyObject *a, PyObject *b)
{
    PyObject *self = a;
    PyObject *args = b;

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        // TODO this is illegal throw an error
        return NULL;
    }
    else
    {

        int device = getDevice(((TensorObject *)a)->matrix, ((TensorObject *)b)->matrix);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);

        if (device == 0)
        {
            obj->matrix = MatMul(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);
        }
        else if (device == 1)
        {
            obj->matrix = GpuMatrixMatMul(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);
        }
        else
        {
            return NULL;
        }

        return (PyObject *)obj;
    }
}

static PyObject *
tensor_isEqual(PyObject *a, PyObject *b)
{
    PyObject *self = a;
    PyObject *args = b;

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    return PyBool_FromLong(isEqual(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix));
}

static PyObject *
tensor_cuda(TensorObject *self)
{
    if (self->matrix->device != 1)
    {
        sendToGpu(self->matrix);
    }
    return (PyObject *)self;
}

static PyObject *
tensor_host(TensorObject *self)
{
    if (self->matrix->device != 0)
    {
        sendToHost(self->matrix);
    }

    return (PyObject *)self;
}
