// https://docs.python.org/3/extending/newtypes_tutorial.html

#include <Python.h>
#include "../cpu/shared.h";

typedef struct
{
    PyObject_HEAD int size;
    Matrix *matrix;
    /*
        Here we store the
    */
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
    printf("pointer bro = %p\n", self);
    printf("%i\n", self->size);

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

PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    /*    subtype->matrix = createMatrix(4, 4);
        fill(subtype->matrix, 1);
        subtype->size = 42;*/
    return subtype;
}

static int *init_object(TensorObject *self, PyObject *Py_UNUSED(ignored))
{
    //  subtype->matrix = createMatrix(4, 4);
    printf("%p", self);

    return self;
}

static int
Noddy_init(TensorObject *self, PyObject *args, PyObject *kwds)
{
    /*  PyObject *first=NULL, *last=NULL, *tmp;

      static char *kwlist[] = {"first", "last", "number", NULL};

      if (! PyArg_ParseTupleAndKeywords(args, kwds, "|SSi", kwlist,
                                        &first, &last,
                                        &self->number))
          return -1;
  */
    /*
        if (first) {
            tmp = self->first;
            Py_INCREF(first);
            self->first = first;
            Py_DECREF(tmp);
        }

        if (last) {
            tmp = self->last;
            Py_INCREF(last);
            self->last = last;
            Py_DECREF(tmp);
        }
    */
    return 0;
}

void tp_free(TensorObject *self)
{
    printf("%p", self);
    freeMatrix(self->matrix);
}

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

static int
Custom_init(TensorObject *self, PyObject *args, PyObject *kwds)
{
    printf("%p Brooo!!! we out here in new !!\n", self);

    return 0;
}

static PyTypeObject TensorType = {
    PyObject_HEAD_INIT(NULL)
    //.ob_size = 0,
    .tp_name = "tensor",
    .tp_doc = PyDoc_STR("Tensor objects"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = sizeof(Matrix),
    //.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE ,
    .tp_new = Custom_new,
    .tp_init = Custom_init,
    //  .tp_init = Noddy_init,
    // .tp_free= tp_free,
    //.tp_alloc = init_object,
    .tp_as_number = &magic_num_methods,
    .tp_methods = tensor_methods,
   // .tp_init = (initproc)init_object,
};

static PyObject *
tensor_add(PyObject *self, PyObject *args)
{
    printf("we out here\n");

  if (PyType_Ready(&TensorType)) {
    return NULL;
  } 

    /* Pass two arguments, a string and an int. */
    PyObject *argList = Py_BuildValue("si", "hello", 42);
   /* Call the class object. */
   printf("CREATING IT RO\n");
    TensorObject *obj = PyObject_CallObject((PyObject *) &TensorType, argList);
    Py_INCREF(obj);

    printf("out it now %p\n", obj);
    printf("New matrix created ? %p \n ", obj);
    obj->matrix = createMatrix(4, 4);
    printf("yeah boiii\n");

    return obj;
}


