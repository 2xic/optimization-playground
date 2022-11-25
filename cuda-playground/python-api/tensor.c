// https://docs.python.org/3/extending/newtypes_tutorial.html

#include <Python.h>
#include "../cpu/shared.h"

typedef struct
{
    PyObject_HEAD int size;
    Matrix *matrix;
} TensorObject;

static TensorObject *tensor_add(PyObject *self, PyObject *args);
static TensorObject *tensor_mul(PyObject *self, PyObject *args);
static TensorObject *tensor_subtract(PyObject *self, PyObject *args);

static TensorObject *Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Ones(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Rand(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Print(TensorObject *self, PyObject *Py_UNUSED(ignored));


void tp_dealloc(TensorObject *self);
void tp_free(void *self);

PyNumberMethods magic_num_methods = {
    .nb_add = tensor_add,
    .nb_multiply = tensor_mul,
    .nb_subtract = tensor_subtract};

static PyMethodDef tensor_methods[] = {
    {"zeros", (PyCFunction)Zeros, METH_NOARGS, "Creates a zero tensor"},
    {"ones", (PyCFunction)Ones, METH_NOARGS, "Creates a ones tensor"},
    {"rand", (PyCFunction)Rand, METH_NOARGS, "Creates a random tensor"},
    {"print", (PyCFunction)Print, METH_NOARGS,
     "Print the tensor"},
    {NULL} /* Sentinel */
};

static PyTypeObject TensorType = {
    PyObject_HEAD_INIT(NULL)
        .tp_name = "tensor",
    .tp_doc = PyDoc_STR("Tensor objects"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = sizeof(Matrix),
    .tp_new = PyType_GenericNew,
    .tp_free = tp_free,
    //  .tp_dealloc=tp_dealloc,
    .tp_as_number = &magic_num_methods,
    .tp_methods = tensor_methods,
};


void tp_dealloc(TensorObject *self)
{
    //    free(self->matrix);
    printf("deallocate +\n");
}

void tp_free(void *self)
{
    printf("free the beef? +\n");
}

static TensorObject *
Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    return self;
}

static TensorObject *
Ones(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    fill(self->matrix, 1);
    self->size = 42;
    return self;
}

static TensorObject *
Rand(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    fillRandom(self->matrix);
    return self;
}

static TensorObject *
Print(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("Matrix pointer = %p\n", self);

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

    return self;
}

static TensorObject *
tensor_subtract(PyObject *self, PyObject *args)
{

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        long value = PyLong_AsLong(args);
        printf("Add a constant value :) %li\n", value);
        return self;
    }
    else
    {
        TensorObject *obj = (TensorObject*)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Add(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}

static TensorObject *
tensor_add(PyObject *self, PyObject *args)
{

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        long value = PyLong_AsLong(args);
        printf("Add a constant value :) %li\n", value);
        return self;
    }
    else
    {
        TensorObject *obj = (TensorObject*)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Add(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}

static TensorObject *
tensor_mul(PyObject *self, PyObject *args)
{

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }
    printf("Tensor mul :)\n");

    if (PyLong_Check(args))
    {
        long value = PyLong_AsLong(args);
        printf("Add a constant value :) %li\n", value);
        return self;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Add(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}
