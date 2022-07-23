/*
 * Copyright 2022, Micah Thornton and Chanhee Park <parkchanhee@gmail.com>
 *
 * This file is part of GEMMULEM
 *
 * GEMMULEM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GEMMULEM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GEMMULEM.  If not, see <http://www.gnu.org/licenses/>.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdarg.h>

#include "EM.h"

#ifdef DEBUG
#define DEBUGLOG(fmt, ...) do { fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__);  } while(0) 
#else
#define DEBUGLOG(fmt, ...) 
#endif

/**
 * create double array.array with length.
 * 
 */
static PyObject* create_darray_object(size_t length)
{
    const char* array_func_name = "array";

    PyObject* pArrayModule = PyImport_ImportModule("array");
    PyObject* pValue = NULL;

    if (!pArrayModule) {
        DEBUGLOG("Can't load array module\n");
        return NULL;
    }

    // get array class constructor
    PyObject* pArrayClass = PyObject_GetAttrString(pArrayModule, array_func_name);

    if (pArrayClass && PyCallable_Check(pArrayClass)) {
        PyObject* pDummyList = PyList_New(length);
        PyObject* pZero = PyFloat_FromDouble(0.0);
        for(size_t i = 0; i < length; i++) {
            PyList_SET_ITEM(pDummyList, i, pZero);
        }

        PyObject* pArgs = PyTuple_New(2);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString("d"));
        PyTuple_SetItem(pArgs, 1, pDummyList);

        pValue = PyObject_CallObject(pArrayClass, pArgs);

        if (pValue == NULL) {
            PyErr_Print();
        }

        Py_DECREF(pArgs);
        Py_DECREF(pDummyList);
    } else {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        DEBUGLOG("Cannot find function \"%s\"\n", array_func_name);
    }

    Py_XDECREF(pArrayClass);
    Py_DECREF(pArrayModule);

    return pValue;
}


/**
 * gemmulem.expectationmaximization(compatibilitymatrix, counts, verbose, maxiter, rtole)
 *
 * @return list
 *
 */
static PyObject* pygemmulem_em(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"", "", "verbose", "maxiter", "rtole", NULL};
    PyObject* compobj = NULL;
    PyObject* countobj = NULL;

    int verbose = 0;
    int maxiter = 1000;
    double rtole = 0.00001;

    int NumPattern = 0;
    int NumCategory = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|iid:expectationmaximization", kwlist,
                &compobj, &countobj, &verbose, &maxiter, &rtole))
    {
        DEBUGLOG("Can't parse argument\n");
        return NULL;
    }

    DEBUGLOG("verbose: %d\n", verbose);
    DEBUGLOG("maxiter: %d\n", maxiter);
    DEBUGLOG("rtole: %f\n", rtole);

    // compobj is list of str
    // check type

    if (!PyList_Check(compobj)) {
        DEBUGLOG("invalid compobj type\n");
        return NULL;
    }
    if (!PyList_Check(countobj)) {
        DEBUGLOG("invalid countobj type\n");
        return NULL;
    }

    NumPattern = PyList_Size(compobj);
    DEBUGLOG("numpattern: %d\n", NumPattern);

    if (PyList_Size(countobj) != NumPattern) {
        DEBUGLOG("Invalid countobj size\n");
        return NULL;
    }

    if (NumPattern > 0) {
        PyObject *item = PyList_GetItem(compobj, 0);

        const char *itemstr = PyUnicode_AsUTF8(item);
        if (itemstr) {
            NumCategory = strlen(itemstr);
        }

#if 0
        PyObject_Print(item, stdout, 0);
        fprintf(stdout, "\n");
#endif
    }

    DEBUGLOG("numcategory: %d\n", NumCategory);

    int* CountPtr = (int*)malloc(sizeof(int) * NumPattern);
    char* CompatMatrixPtr = (char*)malloc(NumCategory * NumPattern);

    memset(CountPtr, 0, sizeof(int) * NumPattern);
    memset(CompatMatrixPtr, 0, NumCategory * NumPattern);

    for(int i = 0; i < NumPattern; i++) {
        PyObject *item = PyList_GetItem(compobj, i);

        const char *itemstr = PyUnicode_AsUTF8(item);

        if (strlen(itemstr) != NumCategory) {
            DEBUGLOG("Invalid compatibility matrix\n");
            free(CompatMatrixPtr);
            free(CountPtr);
            return NULL;
        }

        memcpy(CompatMatrixPtr + i * NumCategory, itemstr, NumCategory);

        item = PyList_GetItem(countobj, i);
        CountPtr[i] = PyLong_AsLong(item);
    }


    EMConfig_t cfg;
    EMResult_t result;

    cfg.verbose = verbose;
    cfg.maxiter = maxiter;
    cfg.rtole = rtole;

    ExpectationMaximization(
            CompatMatrixPtr,
            NumPattern, /* NumRows */
            NumCategory, /* NumCols */
            CountPtr,
            NumPattern, /* NumCount */
            &result, &cfg);
   
    if (result.size != NumCategory) {
        DEBUGLOG("Invalid result size\n");
    }

    // preparing return object
    PyObject* listobj = PyList_New(result.size);
    for(int i = 0; i < result.size; i++) {
        PyList_SET_ITEM(listobj, i, PyFloat_FromDouble(result.values[i]));
    }

    ReleaseEMResult(&result);

    free(CountPtr);
    free(CompatMatrixPtr);

    return listobj;
}


/**
 * gemmulem.unmixexponentials(values, numexponentials, verbose, maxiter, rtole)
 *
 * @return list
 *
 */
static PyObject* pygemmulem_unmixexponentials(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"", "", "verbose", "maxiter", "rtole", NULL};
    PyObject* valueobj = NULL;
    int num_exponentials = 0;
    int verbose = 0;
    int maxiter = 1000;
    double rtole = 0.000001;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|iid:unmixexponentials", kwlist,
                &valueobj, &num_exponentials, &verbose, &maxiter, &rtole)) {
        DEBUGLOG("Can't parse argument\n");
        return NULL;
    }

    DEBUGLOG("num_exponentials: %d\n", num_exponentials);
    DEBUGLOG("verbose: %d\n", verbose);
    DEBUGLOG("maxiter: %d\n", maxiter);
    DEBUGLOG("rtole: %f\n", rtole);


    // create bufferview to value object 
    if (!PyObject_CheckBuffer(valueobj)) {
        DEBUGLOG("Doesn't support buffer object\n");
        // FIXME: TypeError?
        Py_RETURN_NONE;
    }

    Py_buffer bufferview;
    if (PyObject_GetBuffer(valueobj, &bufferview, PyBUF_SIMPLE) < 0) {
        DEBUGLOG("Can't get buffer\n");
        Py_RETURN_NONE;
    }

    size_t len = bufferview.len / bufferview.itemsize;
    double *valueptr = (double *)bufferview.buf;

    EMConfig_t cfg;
    EMResultExponential_t result;

    cfg.verbose = verbose;
    cfg.maxiter = maxiter;
    cfg.rtole = rtole;

    UnmixExponentials(valueptr, len, num_exponentials, &result, &cfg);

    PyBuffer_Release(&bufferview);


    // preparing return object
    int resultlen = result.numExponentials * 2; /* means, probs */
    PyObject* array = create_darray_object(resultlen);

    Py_buffer resultview;
    if (PyObject_GetBuffer(array, &resultview, PyBUF_WRITABLE) < 0) {
        DEBUGLOG("Can't get resultivew\n");
        ReleaseEMResultExponential(&result);
        Py_RETURN_NONE;
    }

    double* resultptr = (double *)resultview.buf;
    memcpy(resultptr, result.means_final, sizeof(double) * result.numExponentials);
    memcpy(resultptr + result.numExponentials, result.probs_final, sizeof(double) * result.numExponentials);

    PyBuffer_Release(&resultview);
    ReleaseEMResultExponential(&result);

    if (array) {
        return array;
    }
    Py_RETURN_NONE;
}

/**
 * gemmulem.unmixgaussians(values, numgaussians, verbose, maxiter, rtole)
 *
 * @return list
 *
 */
static PyObject* pygemmulem_unmixgaussians(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"", "", "verbose", "maxiter", "rtole", NULL};
    PyObject* values = NULL;
    int num_gaussians = 0; 
    int verbose = 0;
    int maxiter = 1000;
    double rtole = 0.000001;

#if 0
    PyObject_Print(kwargs, stdout, 0);
    fprintf(stdout, "\n");
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|iid:unmixgaussians", kwlist,
                &values, &num_gaussians, &verbose, &maxiter, &rtole))
    {
        DEBUGLOG("Can't parse argument\n");
        return NULL;
    }

    DEBUGLOG("num_gaussians: %d\n", num_gaussians); 
    DEBUGLOG("verbose: %d\n", verbose);
    DEBUGLOG("maxiter: %d\n", maxiter);
    DEBUGLOG("rtole: %f\n", rtole);

    // using Buffer Protocol
    //
    if (!PyObject_CheckBuffer(values)) {
        DEBUGLOG("doesn't support buffer object\n");
        Py_RETURN_NONE;
    }

    Py_buffer bufferview;

    if (PyObject_GetBuffer(values, &bufferview, PyBUF_SIMPLE) < 0) {
        DEBUGLOG("Can't get buffer\n");
        Py_RETURN_NONE;
    }

    size_t len = bufferview.len / bufferview.itemsize;
    double* valueptr = (double *)bufferview.buf;

    EMConfig_t cfg;
    EMResultGaussian_t result;

    cfg.verbose = verbose;
    cfg.maxiter = maxiter;
    cfg.rtole = rtole;

    UnmixGaussians(valueptr, len, num_gaussians, &result, &cfg);

    PyBuffer_Release(&bufferview);

    // preparing return object
    int result_len = result.numGaussians * 3; /* means, vars, probs */ 
    PyObject* array = create_darray_object(result_len);

    Py_buffer resultview;
    if (PyObject_GetBuffer(array, &resultview, PyBUF_WRITABLE) < 0) {
        DEBUGLOG("Can't get resultview\n");
        ReleaseEMResultGaussian(&result);
        Py_RETURN_NONE;
    }

    double* resultptr = (double *)resultview.buf;
    memcpy(resultptr, result.means_final, sizeof(double) * result.numGaussians);
    memcpy(resultptr + result.numGaussians, result.vars_final, sizeof(double) * result.numGaussians);
    memcpy(resultptr + result.numGaussians*2, result.probs_final, sizeof(double) * result.numGaussians);

    PyBuffer_Release(&resultview);

    ReleaseEMResultGaussian(&result);
    if (array) {
        return array;
    }
    Py_RETURN_NONE;
}

static PyMethodDef pygemmulem_method[] = {
    { "expectationmaximization", (PyCFunction)pygemmulem_em, METH_VARARGS | METH_KEYWORDS, "Expectation Maximization" },
    { "unmixgaussians", (PyCFunction)pygemmulem_unmixgaussians, METH_VARARGS | METH_KEYWORDS, "Unmix Gaussian" },
    { "unmixexponentials", (PyCFunction)pygemmulem_unmixexponentials, METH_VARARGS | METH_KEYWORDS, "Unmix Exponentials" },

	/* */
	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef pygemmulem_module = {
    PyModuleDef_HEAD_INIT,
    "pygemmulem",
    NULL,
    -1,
    pygemmulem_method
};

PyMODINIT_FUNC
PyInit_pygemmulem(void)
{
    return PyModule_Create(&pygemmulem_module);
}
