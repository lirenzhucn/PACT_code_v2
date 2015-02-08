#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// implementation function
void recon_loop_imp(const npy_double *pa_data,
		    const npy_uint64 *idxAll,
		    const npy_double *angularWeight,
		    int nPixelx, int nPixely, int nSteps,
		    int nTimeSamples,
		    npy_double *pa_img) {
  int iStep, y, x, icount, pcount, iskip;

  icount = 0;
  for (iStep=0; iStep<nSteps; iStep++) {
    pcount = 0;
    iskip = nTimeSamples * iStep - 1;
    /* iskip = 1301 * iStep - 1; */
    for (y=0; y<nPixely; y++) {
      for (x=0; x<nPixelx; x++) {
	pa_img[pcount++] +=
	  pa_data[(int)idxAll[icount]+iskip] *
	  angularWeight[icount];
	icount++;
      }
    }
  }
}

// interface function
// inputs:
//   pa_data: numpy.ndarray, ndim=2, dtype=numpy.double
//   idxAll: numpy.ndarray, ndim=3, dtype=numpy.uint
//   angularWeight: numpy.ndarray, ndim=3, dtype=numpy.double
//   nPixelx: int
//   nPixely: int
//   nSteps: int
// output:
//   pa_img: numpy.ndarray, ndim=2, dtype=numpy.double
static PyObject* recon_loop(PyObject* self, PyObject* args) {
  PyArrayObject *p_pa_data, *p_idxAll, *p_angularWeight;
  int nPixelx, nPixely, nSteps;
  PyObject *p_pa_img;
  npy_intp dim_pa_img[2];

  int paDataValid, idxAllValid, angularWeightValid;
  npy_double *pa_data, *angularWeight;
  npy_uint64 *idxAll;
  npy_double *pa_img;

  // extract argument tuple
  if (!PyArg_ParseTuple(args, "O!O!O!iii",
  			&PyArray_Type, &p_pa_data,
  			&PyArray_Type, &p_idxAll,
  			&PyArray_Type, &p_angularWeight,
  			&nPixelx, &nPixely, &nSteps)) {
    return Py_None;
  }

  // extract and validate variables
  paDataValid = (PyArray_ISFLOAT(p_pa_data)) &&
    (PyArray_CHKFLAGS(p_pa_data, NPY_ARRAY_FARRAY));
  idxAllValid = (PyArray_ISUNSIGNED(p_idxAll)) &&
    (PyArray_CHKFLAGS(p_idxAll, NPY_ARRAY_FARRAY));
  angularWeightValid = (PyArray_ISFLOAT(p_angularWeight)) &&
    (PyArray_CHKFLAGS(p_angularWeight, NPY_ARRAY_FARRAY));
  if (!paDataValid || !idxAllValid || !angularWeightValid) {
    printf("%d, %d, %d\n", paDataValid, idxAllValid, angularWeightValid);
    goto fail;
  }
  dim_pa_img[0] = nPixely;
  dim_pa_img[1] = nPixelx;
  p_pa_img = PyArray_ZEROS(2, dim_pa_img, NPY_DOUBLE, 1);
  pa_data = (npy_double *)PyArray_DATA(p_pa_data);
  idxAll = (npy_uint64 *)PyArray_DATA(p_idxAll);
  angularWeight = (npy_double *)PyArray_DATA(p_angularWeight);
  pa_img = (npy_double *)PyArray_DATA(p_pa_img);

  // call implementation function
  recon_loop_imp(pa_data, idxAll, angularWeight,
		 nPixelx, nPixely, nSteps,
		 PyArray_SHAPE(p_pa_data)[0],
		 pa_img);

  // return value
  return p_pa_img;
  // failed situation
 fail:
  return Py_None;
}

npy_double round(npy_double d) {
  return (d>0.0 ? floor(d+0.5) : floor(d-0.5));
}

void find_index_map_and_angular_weight_imp
(const int nSteps, const npy_double *xImg, const npy_double *yImg,
 const npy_double *xReceive, const npy_double *yReceive, const npy_double *delayIdx,
 const npy_double vm, const npy_double fs, const long nSize2D,
 npy_uint64 *idxAll, npy_double *angularWeight, npy_double *totalAngularWeight) {
  /* Reference python codes
def find_index_map_and_angular_weight\
    (nSteps, xImg, yImg, xReceive, yReceive, delayIdx, vm, fs):
    totalAngularWeight = np.zeros(xImg.shape, order='F')
    idxAll = np.zeros((xImg.shape[0], xImg.shape[1], nSteps),\
                      dtype=np.uint, order='F')
    angularWeight = np.zeros((xImg.shape[0], xImg.shape[1], nSteps),\
                             order='F')
    for n in range(nSteps):
        r0 = np.sqrt(np.square(xReceive[n]) + np.square(yReceive[n]))
        dx = xImg - xReceive[n]
        dy = yImg - yReceive[n]
        rr0 = np.sqrt(np.square(dx) + np.square(dy))
        cosAlpha = np.abs((-xReceive[n]*dx-yReceive[n]*dy)/r0/rr0)
        cosAlpha = np.minimum(cosAlpha, 0.999)
        angularWeight[:,:,n] = cosAlpha/np.square(rr0)
        totalAngularWeight = totalAngularWeight + angularWeight[:,:,n]
        idx = np.around((rr0/vm - delayIdx[n]) * fs)
        idxAll[:,:,n] = idx
    return (idxAll, angularWeight, totalAngularWeight)
  */

  int n, i;
  npy_double r0, rr0, dx, dy, cosAlpha;
  for (n=0; n<nSteps; n++) {
    r0 = sqrt(xReceive[n]*xReceive[n] + yReceive[n]*yReceive[n]);
    for (i=0; i<nSize2D; i++) {
      dx = xImg[i] - xReceive[n];
      dy = yImg[i] - yReceive[n];
      rr0 = sqrt(dx*dx + dy*dy);
      cosAlpha = fabs((-xReceive[n]*dx-yReceive[n]*dy)/r0/rr0);
      cosAlpha = cosAlpha<0.999 ? cosAlpha : 0.999;
      angularWeight[n*nSize2D+i] = cosAlpha / (rr0 * rr0);
      totalAngularWeight[i] += angularWeight[n*nSize2D+i];
      idxAll[n*nSize2D+i] = (npy_uint64)round((rr0/vm - delayIdx[n]) * fs);
    }
  }
}

// speed-up implementation of find_index_map_and_angular_weight
// inputs:
//   nSteps: int
//   xImg: ndarray, ndim=2, dtype=double, same size as the 2d image
//   yImg: same as xImg
//   xReceive: ndarray, ndim=1, dtype=double, length=nSteps
//   yReceive: same as xReceive
//   delayIdx: ndarray, ndim=1, dtype=double, length=nSteps
//   vm: double scalar
//   fs: double scalar
// outputs:
//   idxAll: ndarray, ndim=3, dtype=uint,
//           size=[nPixelx,nPixely,nSteps]
//   angularWeight: same as idxAll, except dtype=double
//   totalAngularWeight: same as angularWeight, except 2D
static PyObject* find_index_map_and_angular_weight(PyObject* self, PyObject* args) {
  PyArrayObject *p_xImg, *p_yImg, *p_xReceive, *p_yReceive, *p_delayIdx;
  // PyArrayObject *p_idxAll, *p_angularWeight, *p_totalAngularWeight;
  PyObject *p_idxAll, *p_angularWeight, *p_totalAngularWeight;
  int nSteps;
  npy_double vm, fs;
  npy_double *xImg, *yImg, *xReceive, *yReceive, *delayIdx;
  npy_uint64 *idxAll;
  npy_double *angularWeight, *totalAngularWeight;

  PyObject *returnTuple = PyTuple_New(3);

  /* int xImgValid, yImgValid, xReceiveValid, yReceiveValid, delayIdxValid; */
  npy_intp dim_3d[3];
  npy_intp dim_2d[2];

  // extract argument tuple
  if (!PyArg_ParseTuple(args, "iO!O!O!O!O!dd",
			&nSteps,
			&PyArray_Type, &p_xImg,
			&PyArray_Type, &p_yImg,
			&PyArray_Type, &p_xReceive,
			&PyArray_Type, &p_yReceive,
			&PyArray_Type, &p_delayIdx,
			&vm, &fs)) {
    goto fail;
  }

  // TODO: validate array objects

  // extract data from array objects
  xImg = (npy_double *)PyArray_DATA(p_xImg);
  yImg = (npy_double *)PyArray_DATA(p_yImg);
  xReceive = (npy_double *)PyArray_DATA(p_xReceive);
  yReceive = (npy_double *)PyArray_DATA(p_yReceive);
  delayIdx = (npy_double *)PyArray_DATA(p_delayIdx);
  // building output array objects
  dim_3d[0] = PyArray_SHAPE(p_xImg)[0];
  dim_3d[1] = PyArray_SHAPE(p_xImg)[1];
  dim_3d[2] = nSteps;
  dim_2d[0] = PyArray_SHAPE(p_xImg)[0];
  dim_2d[1] = PyArray_SHAPE(p_xImg)[1];
  p_idxAll = PyArray_ZEROS(3, dim_3d, NPY_UINT64, 1);
  p_angularWeight = PyArray_ZEROS(3, dim_3d, NPY_DOUBLE, 1);
  p_totalAngularWeight = PyArray_ZEROS(2, dim_2d, NPY_DOUBLE, 1);
  idxAll = (npy_uint64 *)PyArray_DATA(p_idxAll);
  angularWeight = (npy_double *)PyArray_DATA(p_angularWeight);
  totalAngularWeight = (npy_double *)PyArray_DATA(p_totalAngularWeight);

  // Call the implementation
  find_index_map_and_angular_weight_imp
    (nSteps, xImg, yImg, xReceive, yReceive, delayIdx,
     vm, fs, dim_2d[0] * dim_2d[1],
     idxAll, angularWeight, totalAngularWeight);

  // return results
  PyTuple_SET_ITEM(returnTuple, 0, (PyObject*)p_idxAll);
  PyTuple_SET_ITEM(returnTuple, 1, (PyObject*)p_angularWeight);
  PyTuple_SET_ITEM(returnTuple, 2, (PyObject*)p_totalAngularWeight);
  return returnTuple;

  // failed situation
 fail:
  PyTuple_SET_ITEM(returnTuple, 0, Py_None);
  PyTuple_SET_ITEM(returnTuple, 1, Py_None);
  PyTuple_SET_ITEM(returnTuple, 2, Py_None);
  return returnTuple;
}


// module initialization codes
// modified to work with both Python 2 and 3

struct module_state {
  PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject * error_out(PyObject *m) {
  struct module_state *st = GETSTATE(m);
  PyErr_SetString(st->error, "something bad happened");
  return NULL;
}

static PyMethodDef ReconMethods[] = {
  {"recon_loop", recon_loop, METH_VARARGS, "Reconstruction loop"},
  {"find_index_map_and_angular_weight", find_index_map_and_angular_weight,
   METH_VARARGS, "Find index map and angular weights for back-projection"},
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {NULL, NULL, 0, NULL} // the end
};

#if PY_MAJOR_VERSION >= 3

static int recon_loop_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int recon_loop_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "recon_loop",
  NULL,
  sizeof(struct module_state),
  ReconMethods,
  NULL,
  recon_loop_traverse,
  recon_loop_clear,
  NULL
};

#endif

#if PY_MAJOR_VERSION >= 3
#define INITERROR return NULL
#else
#define INITERROR return
#endif


// module initialization
PyObject *initializeModule(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("recon_loop", ReconMethods);
#endif

  if (module == NULL) {
    INITERROR;
  }
  struct module_state *st = GETSTATE(module);
  // IMPORTANT: this must be called
  import_array();
  // always return the object.
  // Python 2 interface will ignore the returned object.
  return module;
}

#if PY_MAJOR_VERSION >= 3
PyObject *PyInit_recon_loop(void) {
  return initializeModule();
}
#else
void initrecon_loop(void) {
  initializeModule();
}
#endif
