#include "ring_pact_speedup.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

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
  npy_double *pa_data, *angularWeight, *idxAll;
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
  idxAllValid = (PyArray_ISFLOAT(p_idxAll)) &&
    (PyArray_CHKFLAGS(p_idxAll, NPY_ARRAY_CARRAY));
  angularWeightValid = (PyArray_ISFLOAT(p_angularWeight)) &&
    (PyArray_CHKFLAGS(p_angularWeight, NPY_ARRAY_CARRAY));
  if (!paDataValid || !idxAllValid || !angularWeightValid) {
    printf("%d, %d, %d\n", paDataValid, idxAllValid, angularWeightValid);
    goto fail;
  }
  dim_pa_img[0] = nPixely;
  dim_pa_img[1] = nPixelx;
  p_pa_img = PyArray_ZEROS(2, dim_pa_img, NPY_DOUBLE, 1);
  pa_data = (npy_double *)PyArray_DATA(p_pa_data);
  idxAll = (npy_double *)PyArray_DATA(p_idxAll);
  angularWeight = (npy_double *)PyArray_DATA(p_angularWeight);
  pa_img = (npy_double *)PyArray_DATA((PyArrayObject *)p_pa_img);

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

/**
 * interface function
 * inputs:
 *   paData: ndarray, ndim=3, nSamples*nSteps*zSteps, dtype=double
 *   idxAll: ndarray, ndim=3, nPixely*nPixelx*nSteps, dtype=uint64
 *   angularWeight: ndarray, ndim=3, nPixely*nPixelx*nSteps, dtype=double
 *   totalAngularWeight: ndarray, ndim=2, nPixely*nPixelx, dtype=double
 * output:
 *   paImg: ndarray, ndim=3, nPixely*nPixelx*zSteps, dtype=double
 */
static PyObject *backproject_loop(PyObject *self, PyObject *args) {
  // input/output arguments
  PyArrayObject *p_paData, *p_idxAll, *p_angularWeight, *p_totalAngularWeight;
  PyObject *p_paImg;
  // input/output data pointers
  npy_double *paData, *angularWeight, *totalAngularWeight, *idxAll;
  npy_double *paImg;
  // size variables to be extracted from arrays
  int nPixelx, nPixely, zSteps, nSteps, nSamples;
  // boolean flags for data validation
  int paDataValid, idxAllValid, angularWeightValid, totalAngularWeightValid;
  // size arrays
  npy_intp dim_paImg[3];

  // extract argument tuple
  if (!PyArg_ParseTuple(args, "O!O!O!O!",
        &PyArray_Type, &p_paData,
        &PyArray_Type, &p_idxAll,
        &PyArray_Type, &p_angularWeight,
        &PyArray_Type, &p_totalAngularWeight)) {
    printf("Error: Something wrong with input arguments.\n");
    return Py_None;
  }

  // extract and validate variables
  nSamples = PyArray_SHAPE(p_paData)[0];
  nSteps = PyArray_SHAPE(p_paData)[1];
  zSteps = PyArray_SHAPE(p_paData)[2];
  nPixely = PyArray_SHAPE(p_idxAll)[0];
  nPixelx = PyArray_SHAPE(p_idxAll)[1];
  nSteps = PyArray_SHAPE(p_idxAll)[2];
  paDataValid = (PyArray_ISFLOAT(p_paData)) &&
    (PyArray_CHKFLAGS(p_paData, NPY_ARRAY_FARRAY));
  idxAllValid = (PyArray_ISFLOAT(p_idxAll)) &&
    (PyArray_CHKFLAGS(p_idxAll, NPY_ARRAY_CARRAY));
  angularWeightValid = (PyArray_ISFLOAT(p_angularWeight)) &&
    (PyArray_CHKFLAGS(p_angularWeight, NPY_ARRAY_CARRAY));
  totalAngularWeightValid = (PyArray_ISFLOAT(p_totalAngularWeight)) &&
    (PyArray_CHKFLAGS(p_totalAngularWeight, NPY_ARRAY_FARRAY));
  if (!paDataValid || !idxAllValid || !angularWeightValid || !totalAngularWeightValid) {
    printf("%d, %d, %d, %d\n",
        paDataValid, idxAllValid, angularWeightValid, totalAngularWeightValid);
    return Py_None;
  }
  paData = (npy_double *)PyArray_DATA(p_paData);
  idxAll = (npy_double *)PyArray_DATA(p_idxAll);
  angularWeight = (npy_double *)PyArray_DATA(p_angularWeight);
  totalAngularWeight = (npy_double *)PyArray_DATA(p_totalAngularWeight);
  // create paImg object
  dim_paImg[0] = nPixely;
  dim_paImg[1] = nPixelx;
  dim_paImg[2] = zSteps;
  p_paImg = PyArray_ZEROS(3, dim_paImg, NPY_DOUBLE, 1);
  paImg = (npy_double *)PyArray_DATA((PyArrayObject *)p_paImg);

  // call implementation
  backproject_loop_imp(paData, idxAll, angularWeight, totalAngularWeight,
      nPixelx, nPixely, zSteps, nSteps, nSamples, paImg);

  return p_paImg;
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
  npy_double *xImg, *yImg, *xReceive, *yReceive, *delayIdx, *idxAll;
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
  /*p_idxAll = PyArray_ZEROS(3, dim_3d, NPY_UINT64, 0);*/
  p_idxAll = PyArray_ZEROS(3, dim_3d, NPY_DOUBLE, 0);
  p_angularWeight = PyArray_ZEROS(3, dim_3d, NPY_DOUBLE, 0);
  p_totalAngularWeight = PyArray_ZEROS(2, dim_2d, NPY_DOUBLE, 1);
  idxAll = (npy_double *)PyArray_DATA((PyArrayObject *)p_idxAll);
  angularWeight = (npy_double *)PyArray_DATA((PyArrayObject *)p_angularWeight);
  totalAngularWeight = (npy_double *)PyArray_DATA((PyArrayObject *)p_totalAngularWeight);

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

/**
 * interface function: daq_loop
 * inputs:
 *   packData1: numpy.ndarray, ndim=2, dtype=uint16, 2*dataBlockSize x packSize
 *   packData2: same as packData1
 *   chanMap: numpy.ndarray, ndim=1, dtype=uint32
 *   numExperiments: int
 * output:
 *   chndata: numpy.ndarray, ndim=2, dtype=double, DataBlockSize x NumElements
 *   chndata_all: numpy.ndarray, ndim=2, dtype=double,
 *     DataBlockSize x NumElements*numExperiments
 */
static PyObject *daq_loop(PyObject *self, PyObject *args) {
  // definitions
  PyArrayObject *p_packData1, *p_packData2, *p_chanMap;
  PyArrayObject *p_packData1_copy, *p_packData2_copy;
  PyObject *p_chndata, *p_chndata_all;
  npy_intp dim_chndata[2];
  npy_intp dim_chndata_all[2];
  int numExperiments;
  int packSize;
  npy_uint32 *packData1;
  npy_uint32 *packData2;
  npy_uint32 *chanMap;
  npy_double *chndata;
  npy_double *chndata_all;
  // extract arguments
  PyObject *returnTuple = PyTuple_New(2);
  if (!PyArg_ParseTuple(args, "O!O!O!i",
        &PyArray_Type, &p_packData1,
        &PyArray_Type, &p_packData2,
        &PyArray_Type, &p_chanMap,
        &numExperiments)) {
    PyTuple_SET_ITEM(returnTuple, 0, Py_None);
    PyTuple_SET_ITEM(returnTuple, 1, Py_None);
    return returnTuple;
  }
  // copy pack data arrays
  p_packData1_copy = (PyArrayObject *)
    PyArray_ContiguousFromAny((PyObject *)p_packData1, NPY_UINT32, 0, 0);
  p_packData2_copy = (PyArrayObject *)
    PyArray_ContiguousFromAny((PyObject *)p_packData2, NPY_UINT32, 0, 0);
  // extract argument data
  packData1 = (npy_uint32 *)PyArray_DATA(p_packData1_copy);
  packData2 = (npy_uint32 *)PyArray_DATA(p_packData2_copy);
  chanMap = (npy_uint32 *)PyArray_DATA(p_chanMap);
  packSize = PyArray_SHAPE(p_packData1_copy)[1];
  // build output array objects
  dim_chndata[0] = DataBlockSize;
  dim_chndata[1] = NumElements;
  dim_chndata_all[0] = DataBlockSize;
  dim_chndata_all[1] = NumElements*numExperiments;
  p_chndata = PyArray_ZEROS(2, dim_chndata, NPY_DOUBLE, 1);
  p_chndata_all = PyArray_ZEROS(2, dim_chndata_all, NPY_DOUBLE, 1);
  chndata = (npy_double *)PyArray_DATA((PyArrayObject *)p_chndata);
  chndata_all = (npy_double *)PyArray_DATA((PyArrayObject *)p_chndata_all);

  // call the implementation
  daq_loop_imp(packData1, packData2, chanMap, numExperiments, packSize,
      chndata, chndata_all);

  // return results
  PyTuple_SET_ITEM(returnTuple, 0, p_chndata);
  PyTuple_SET_ITEM(returnTuple, 1, p_chndata_all);
  return returnTuple;
}

/**
 * interface function: generateChanMap
 * inputs:
 *   numElement: int
 * outputs:
 *   chanMap: numpy.ndarray, ndim=1, dtype=numpy.uint32
 */
static PyObject *generateChanMap(PyObject *self, PyObject *args) {
  npy_intp numElements;
  PyObject *p_chanMap;
  npy_uint32 *chanMap;
  // extract argument tuple
  if (!PyArg_ParseTuple(args, "i", &numElements)) { return Py_None; }
  p_chanMap = PyArray_ZEROS(1, &numElements, NPY_UINT32, 1);
  chanMap = (npy_uint32 *)PyArray_DATA((PyArrayObject *)p_chanMap);
  // call the actual function
  generateChanMap_imp(numElements, chanMap);
  // return result
  return p_chanMap;
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

static PyMethodDef RingPactSpeedupMethods[] = {
  {"recon_loop", recon_loop, METH_VARARGS, "Reconstruction loop"},
  {"backproject_loop", backproject_loop, METH_VARARGS,
   "Backprojection loop over z"},
  {"find_index_map_and_angular_weight", find_index_map_and_angular_weight,
   METH_VARARGS, "Find index map and angular weights for back-projection"},
  {"generateChanMap", generateChanMap, METH_VARARGS,
   "Generate correct channel map based DAQ indices."},
  {"daq_loop", daq_loop, METH_VARARGS,
   "Fast version of the unpacking function"},
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {NULL, NULL, 0, NULL} // the end
};

#if PY_MAJOR_VERSION >= 3

static int ring_pact_speedup_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int ring_pact_speedup_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "recon_loop",
  NULL,
  sizeof(struct module_state),
  RingPactSpeedupMethods,
  NULL,
  ring_pact_speedup_traverse,
  ring_pact_speedup_clear,
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
  PyObject *module = Py_InitModule("ring_pact_speedup", RingPactSpeedupMethods);
#endif

  if (module == NULL) {
    INITERROR;
  }
  /*struct module_state *st = GETSTATE(module);*/
  // IMPORTANT: this must be called
  import_array();
  // always return the object.
  // Python 2 interface will ignore the returned object.
  return module;
}

#if PY_MAJOR_VERSION >= 3
PyObject *PyInit_ring_pact_speedup(void) {
  return initializeModule();
}
#else
void initring_pact_speedup(void) {
  initializeModule();
}
#endif
