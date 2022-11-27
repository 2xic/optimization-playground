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
static TensorObject *tensor_divide(PyObject *self, PyObject *args);
static TensorObject *tensor_subtract(PyObject *self, PyObject *args);
static TensorObject *tensor_transpose(TensorObject *self);
static TensorObject *tensor_matmul(PyObject *self, PyObject *args);
static TensorObject *tensor_negative(TensorObject *self);
static TensorObject *tensor_isEqual(PyObject *a, PyObject *b);

static TensorObject *Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Ones(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Rand(TensorObject *self, PyObject *Py_UNUSED(ignored));
static TensorObject *Print(TensorObject *self, PyObject *Py_UNUSED(ignored));

static TensorObject *tensor_cuda(TensorObject *self);
static TensorObject *tensor_host(TensorObject *self);

int getDevice(Matrix *a, Matrix *b);

void tp_dealloc(TensorObject *self);
void tp_free(void *self);

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

    return self;
}

static TensorObject *
tensor_subtract(PyObject *a, PyObject *b)
{
    TensorObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction
    );

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = SubtractConstant(((TensorObject *)self)->matrix, value, direction);
        return obj;
    }
    else
    {
        
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Subtract(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}

static TensorObject *
tensor_negative(TensorObject *self)
{
    TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
    obj->matrix = MulConstant(((TensorObject *)self)->matrix, -1, 0);

    return obj;
}

static TensorObject *
tensor_add(PyObject *a, PyObject *b)
{
    TensorObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction
    );


    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = AddConstant(((TensorObject *)self)->matrix, value, direction);

        return obj;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Add(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}

static TensorObject *
tensor_mul(PyObject *a, PyObject *b)
{
    TensorObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction
    );

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = MulConstant(((TensorObject *)self)->matrix, value, direction);

        return obj;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = Mul(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);

        return obj;
    }
}

static TensorObject *
tensor_divide(PyObject *a, PyObject *b)
{
    TensorObject *self;
    PyObject *args;
    int direction;

    getTensorPointer(
        a, b,
        &self, &args, &direction
    );

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        float value = PyFloat_AsDouble(args);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        obj->matrix = DivideConstant(((TensorObject *)self)->matrix, value, direction);

        return obj;
    }
    else
    {
        // TODO
        return NULL;
    }
}

static TensorObject *
tensor_transpose(TensorObject *self)
{

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
    int s[] = {self->matrix->columns, self->matrix->rows};

    Matrix *results = createMatrixN(s, 2);
    obj->matrix = results;

    for (int i = 0; i < self->matrix->rows; i++)
    {
        for (int j = 0; j < self->matrix->columns; j++)
        {
            setElement(obj->matrix, j, i, getElement(self->matrix, i, j));
        }
    }

    return obj;
}

static TensorObject *
tensor_matmul(PyObject *a, PyObject *b)
{
    TensorObject *self = a;
    PyObject *args= b;

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        /*
            This is illegal throw an error
        */
        long value = PyLong_AsLong(args);
        printf("Add a constant value :) %li\n", value);
        return NULL;
    }
    else
    {

        int device = getDevice(((TensorObject*)a)->matrix, ((TensorObject*)b)->matrix);
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);

        if (device == 0) {
            obj->matrix = MatMul(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);
        } else if (device == 1) {
            obj->matrix = MatrixMatMul(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix);
        } else {
            // Throw error
        }
        
        return obj;
    }
}

static TensorObject *
tensor_isEqual(PyObject *a, PyObject *b)
{
    TensorObject *self = a;
    PyObject *args= b;

    if (PyType_Ready(&TensorType))
    {
        return NULL;
    }

    if (PyLong_Check(args))
    {
        /*
            This is illegal throw an error
        */
        long value = PyLong_AsLong(args);
        printf("Add a constant value :) %li\n", value);
        return self;
    }
    else
    {
        TensorObject *obj = (TensorObject *)PyObject_CallObject((PyObject *)&TensorType, NULL);
        return PyBool_FromLong(isEqual(((TensorObject *)self)->matrix, ((TensorObject *)args)->matrix));
    }
}

// One of the objects will always be 
void getTensorPointer(PyObject *a, PyObject *b, PyObject **tensor, PyObject **args, int*direction ) {
    if (PyLong_Check(a)) {
        *tensor = b;
        *args = a; 
        *direction = 1;
    } else {
        *tensor = a;
        *args = b; 
        *direction = 0;
    }
}

int getDevice(Matrix *a, Matrix *b) {
    if (a->device == b->device){
        return a->device;
    } else {
        return -1;
    }
}


static TensorObject *
tensor_cuda(TensorObject *self)
{
    sendToGpu(self->matrix);
    return self;
}

static TensorObject *
tensor_host(TensorObject *self)
{
    sendToHost(self->matrix);

    return self;
}
