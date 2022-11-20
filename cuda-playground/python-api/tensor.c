// https://docs.python.org/3/extending/newtypes_tutorial.html

#include <Python.h>
#include "../cpu/shared.h";

typedef struct {
    PyObject_HEAD
    int size;
    Matrix *matrix;
    /*
        Here we store the 
    */
} TensorObject;

static PyObject *
Zeros(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    self->matrix = createMatrix(4, 4);
    self->size = 42;

    printf("%i\n", self->size);

    return self;
}

static PyObject *
Print(TensorObject *self, PyObject *Py_UNUSED(ignored))
{    
    printf("%i\n", self->size);

    for(int i= 0; i < 4; i++) {
        for(int j = 0; j <4; j++) {
            if (j > 0) {
                printf(", ");
            }
            printf("%i", getElement(self->matrix, i, j));
        }
        printf("\n");
    }

    return self;
}

static PyObject* tensor_add(PyObject* self, PyObject* args);

PyNumberMethods noddy_nums = {
  .nb_add = tensor_add,
};


static PyMethodDef Custom_methods[] = {
    {"zeros", (PyCFunction) Zeros, METH_NOARGS,
     "Creates a zero tensor"
    },
    {"print", (PyCFunction) Print, METH_NOARGS,
     "Print the tensor"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cudaplayground.Tensor",
    .tp_doc = PyDoc_STR("Tensor objects"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_as_number = &noddy_nums,
    .tp_methods = Custom_methods,
};


static PyObject*
tensor_add(PyObject* self, PyObject* args) {
  printf("%i\n", ((TensorObject*)args)->size);

    // self is a
    // args = tensor b
    // append then create new matrix
    PyObject *obj = _PyObject_New((PyObject *) &TensorType);
    TensorObject *tensor = (TensorObject*)obj;
    tensor->matrix = createMatrix(8, 8);
    tensor->size = 44;
    
    return obj;
}
