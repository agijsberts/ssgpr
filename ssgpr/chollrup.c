/*
 * py_choluprk1(mat): python function for rank-1 cholesky update
 *
 * Adaptation of the code published in LHotse by Matthias Seeger. 
 * For further information, see:
 *
 * M. Seeger, Low Rank Updates for the Cholesky Decomposition, Technical Report (2004).
 * http://people.mmci.uni-saarland.de/~mseeger/papers/cholupdate.pdf
 * http://people.mmci.uni-saarland.de/~mseeger/lhotse/index.html
 * https://github.com/mseeger/chollrup/
 *
 * Author: Arjan Gijsberts
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <cblas.h>

#define EPS                 1e-10

//#define DEBUG

#ifdef DEBUG
/*
 * DEBUG HELPER FUNCTIONS
 */

// CLOCK_GETTIME
#include <time.h>
static double _time() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return (double) (time.tv_sec + ((double) time.tv_nsec / 1.0E9));
}
// CLOCK_GETTIME


/*
 * _print_array(data, size) : Print a 1-dimensional array
 */
static void _print_array(double* x, int size) {
	int i;
	printf("array[");
	for(i = 0; i < size; i++) {
		if(i > 0) printf(",");
		printf("%f", x[i]);
	}
	printf("]");
}

#endif

/*
 * isclose(mat): returns true if two values are approximately equal
 */
static inline int isclose(double val, double ref) {
  return (fabs(val - ref) < EPS); 
}

/*
 * isclosetozero(val): returns true if the value is approximately zero
 */
static inline int isclosetozero(double val) {
  return isclose(val, 0.);
}

/*
 * isuppertng(mat): returns true if upper triangular
 */
static int isuppertng(PyArrayObject* mat) {
  // we assume that the matrix is triangular and only check one corner for 
  // a non-zero element. if not triangular or if both zero, then we prefer 
  // upper triangular
  double up = *((double*) PyArray_GETPTR2(mat, 0, mat->dimensions[1] - 1));
  double low = *((double*) PyArray_GETPTR2(mat, mat->dimensions[0] - 1, 0));
  
  if(isclosetozero(up)) {
    if(isclosetozero(low)) return 1;
    else return 0;
  } else return 1;
}

static PyObject* py_choluprk1(PyObject* self, PyObject* args) {
  PyObject* Lo = NULL;       // L (raw python object)
  PyArrayObject* Lao = NULL; // L (contiguous double array)
  double* L = NULL;          // L (array of doubles)
  PyObject* xo = NULL;       // x (raw python object)
  PyArrayObject* xao = NULL; // x (contiguous double array)
  double* x = NULL;          // x (array of doubles)
  int n = 0;                 // length of vectors
  double* work;              // work vector
  double* cvec;              // cvec
  double* svec;              // svec
  double* tbuff;             // buffer
  int stp;                   // step size
  int i;                     // index
  int sz;                    // size
  

#ifdef DEBUG
  int k;

  double start, end;
  start = _time();
#endif

  // read raw object arguments: L, X
  if (!PyArg_ParseTuple(args, "OO", &Lo, &xo)) return NULL;

  if(Lo == NULL || xo == NULL) return NULL;

  // try to load contiguous double array from Lo object
  //Lao = (PyArrayObject*) PyArray_ContiguousFromObject(Lo, PyArray_DOUBLE, 2, 2);
#if NPY_API_VERSION >= 0x0000000c
  Lao = (PyArrayObject*) PyArray_FROMANY(Lo, PyArray_DOUBLE, 2, 2, NPY_ARRAY_INOUT_ARRAY2);
#else
  Lao = (PyArrayObject*) PyArray_FROMANY(Lo, PyArray_DOUBLE, 2, 2, NPY_ARRAY_INOUT_ARRAY);
#endif
  if(Lao == NULL) {
    return NULL;
  }
  // create data pointer for L
  L = (double*) PyArray_DATA(Lao);

  // try to load contiguous double array from xo object
  xao = (PyArrayObject*) PyArray_FROMANY(xo, PyArray_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
  if(xao == NULL) {
    Py_DECREF(Lao);
    return NULL;
  }
  // create data pointer for x
  x = (double*) PyArray_DATA(xao);

  // check whether dimensions of alpha, y and Q match
  n = Lao->dimensions[0];
  
  if(Lao->dimensions[1] != n || xao->dimensions[0] != n) {
    Py_DECREF(Lao);
    Py_DECREF(xao);
    PyErr_SetString(PyExc_ValueError, "dimension of L does not match dimension of x");
    return NULL;
  }

#ifdef DEBUG
  printf("n: %d\n", n);
  printf("x[%8p]: ", x);
  _print_array(x, n);
  printf("\n");
  
  printf("L[%8p]:\n", L);
  for(k = 0; k < n; k++) {
    _print_array((double*) PyArray_GETPTR1(Lao, k), n);
    printf("\n");
  }
  printf("L.isupper: %d\n", isuppertng(Lao));
/*  printf("L.contiguous: %d\n", Lao->flags & NPY_CONTIGUOUS);
  printf("L.aligned: %d\n", Lao->flags & NPY_ALIGNED);
  printf("L.fortran: %d\n", Lao->flags & NPY_FORTRAN);
  printf("L.updateifcopy: %d\n", Lao->flags & NPY_UPDATEIFCOPY);*/
  
  start = _time();
#endif

  work = malloc(n * sizeof(double));
  cvec = malloc(n * sizeof(double));
  svec = malloc(n * sizeof(double));
  cblas_dcopy(n, x, 1, work, 1);
  //for(i = 0; i < n; i++) { cvec[i] = 0.; svec[i] = 0.; }

#ifdef DEBUG
  printf("work[%8p]: ", work);
  _print_array(work, n);
  printf("\n");
#endif

  stp = isuppertng(Lao) ? 1 : n;
  for(i = 0, sz = n, tbuff = L; i < n - 1; i++) {
    if(isclosetozero(*tbuff) && isclosetozero(work[i])) {
#if NPY_API_VERSION >= 0x0000000c
      PyArray_DiscardWritebackIfCopy(Lao);
#endif
      Py_DECREF(Lao);
      Py_DECREF(xao);
      free(work);
      free(svec);
      free(cvec);
      PyErr_SetString(PyExc_ValueError, "choluprk1 in trouble :( (pos:1)");
      return NULL;
    }

    cblas_drotg(tbuff,work+i,&cvec[i],&svec[i]);

    if(*tbuff < 0.0) {
      *tbuff = -(*tbuff);
      cvec[i] = -cvec[i];
      svec[i] = -svec[i];
    } else if(isclosetozero(*tbuff)) {
#if NPY_API_VERSION >= 0x0000000c
      PyArray_DiscardWritebackIfCopy(Lao);
#endif
      Py_DECREF(Lao);
      Py_DECREF(xao);
      free(work);
      free(svec);
      free(cvec);
      PyErr_SetString(PyExc_ValueError, "choluprk1 in trouble :( (pos:2)");
      return NULL;
    }

    sz--;
    cblas_drot(sz, tbuff + stp, stp, work + i + 1, 1, cvec[i], svec[i]);
    tbuff+=(n+1);
  }
  
  if(!isclosetozero(*tbuff) || !isclosetozero(work[n-1])) {
    cblas_drotg(tbuff, work + n - 1, cvec + i, svec + i);

    if(*tbuff < 0.0) {
      *tbuff = -(*tbuff);
      cvec[i] = -cvec[i];
      svec[i] = -svec[i];
    } else if(isclosetozero(*tbuff)) {
#if NPY_API_VERSION >= 0x0000000c
      PyArray_DiscardWritebackIfCopy(Lao);
#endif
      Py_DECREF(Lao);
      Py_DECREF(xao);
      free(work);
      free(svec);
      free(cvec);
      PyErr_SetString(PyExc_ValueError, "choluprk1 in trouble :( (pos:3)");
      return NULL;
    }
  } else {
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(Lao);
#endif
    Py_DECREF(Lao);
    Py_DECREF(xao);
    free(work);
    free(svec);
    free(cvec);
    PyErr_SetString(PyExc_ValueError, "choluprk1 in trouble :( (pos:4)");
    return NULL;
  }


#ifdef DEBUG
  printf("L_end[%8p]:\n", L);
  for(k = 0; k < n; k++) {
    _print_array((double*) PyArray_GETPTR1(Lao, k), n);
    //_print_array(L+(k*n), k);
    //_print_array(L, 1);
    printf("\n");
  }
  start = _time();
#endif



#ifdef DEBUG
  end = _time();
  printf("start: %f; end: %f; passed: %f\n", start, end, end - start);
#endif

  // for now we do not use work, svec or cvec, could easily be wrapped in 
  // pyarrayobject and returned in the tuple
  free(work);
  free(svec);
  free(cvec);

  // decrement references to L and x; prevent memory leak
#if NPY_API_VERSION >= 0x0000000c
  PyArray_ResolveWritebackIfCopy(Lao);
#endif
  Py_DECREF(Lao);
  Py_DECREF(xao);
#if 0
#endif
  Py_RETURN_NONE;  
}




static char py_choluprk1_doc[] = \
  "Rank 1 update of Cholesky decomposition.";

static char module_doc[] = \
  "module chollrup:\n\
  Fast and stable low rank Cholesky updates.\n\
  Based on the implementation of Matthias Seeger.\n\
  ";

// python accessible method definitions
static PyMethodDef chollrup_methods[] = {
  // cholesky rank 1 update
  {"choluprk1", py_choluprk1, METH_VARARGS, py_choluprk1_doc},

  // required ending of the method table
  {NULL, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "chollrup",          /* m_name */
  module_doc,          /* m_doc */
  -1,                  /* m_size */
  chollrup_methods,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_chollrup(void) {
  // register module
  PyObject *module = PyModule_Create(&moduledef);
  
  // import array type for numpy initialization
  import_array();
  
  return module;
}
#else
// init function
PyMODINIT_FUNC initchollrup() {
  // register module
  Py_InitModule3("chollrup", chollrup_methods, module_doc);

  // import array type for numpy initialization
  import_array();
}
#endif

