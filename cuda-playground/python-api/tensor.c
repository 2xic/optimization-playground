// https://docs.python.org/3/extending/newtypes_tutorial.html

#include <Python.h>
#include "../cpu/shared.h";

typedef struct
{
    PyObject_HEAD int size;
    Matrix *matrix;
} TensorObject;

static PyObject *
Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("zeros -> %p\n", self);
    self->matrix = createMatrix(4, 4);
    self->size = 42;

    return self;
}

static PyObject *
Ones(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("ones -> %p\n", self);
    self->matrix = createMatrix(4, 4);
    fill(self->matrix, 1);
    self->size = 42;

    return self;
}

static PyObject *
Print(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("Matrix pointer = %p\n", self);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (j > 0)
            {
                printf(", ");
            }
            printf("%i", getElement(self->matrix, i, j));
        }
        printf("\n");
    }

    return self;
}

static PyObject *tensor_add(PyObject *self, PyObject *args);

PyNumberMethods magic_num_methods = {
    .nb_add = tensor_add,
};

static PyMethodDef tensor_methods[] = {
    {"zeros", (PyCFunction)Zeros, METH_NOARGS, "Creates a zero tensor"},
    {"ones", (PyCFunction)Ones, METH_NOARGS, "Creates a ones tensor"},
    {"print", (PyCFunction)Print, METH_NOARGS,
     "Print the tensor"},
    {NULL} /* Sentinel */
};

static PyObject *
Custom_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    printf("%p Brooo!!! we out here in new !! %p \n", type);

    TensorObject *self;
    printf("alloc\n");
    // TODO: Fix out this
    self = (TensorObject *)type->tp_alloc(type, 256);
    printf("did alloc\n");
    if (self != NULL)
    {   
        printf("%p create it");
    } else {
        printf("got null ??");
    }
    return (PyObject *)self;
}

void tp_dealloc(PyObject *self) {
    printf("deallocate +\n");
}

void tp_free(void *self) {
    printf("free the beef? +\n");
}

static PyTypeObject TensorType = {
    PyObject_HEAD_INIT(NULL)
    .tp_name = "tensor",
    .tp_doc = PyDoc_STR("Tensor objects"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = sizeof(Matrix),
    .tp_new = PyType_GenericNew,
  //  .tp_new = Custom_new,
    .tp_free= tp_free,
    .tp_dealloc=tp_dealloc,

    //.tp_alloc = init_object,
    .tp_as_number = &magic_num_methods,
    .tp_methods = tensor_methods,
};

static PyObject *
tensor_add(PyObject *self, PyObject *args)
{

    if (PyType_Ready(&TensorType)) {
        return NULL;
    } 

    PyObject *argList = Py_BuildValue("si", "hello", 42);
    TensorObject *obj = PyObject_CallObject((PyObject *) &TensorType, argList);
    Py_INCREF(obj);
    obj->matrix = createMatrix(4, 4);

    return obj;
}


