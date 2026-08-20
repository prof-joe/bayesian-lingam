# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""Compiled full optimizer backend for the verified Bayesian-ICA Dijkstra.

This module moves the complete L-BFGS-B iteration loop and the multistart
selection loop into compiled Cython code.  It calls SciPy's low-level compiled
``setulb`` routine directly; ``scipy.optimize.minimize`` and its Python scalar
function wrappers are not used.

The Student-t and affine-ICA likelihood/gradient kernels are evaluated in C
and use BLAS/LAPACK for the dense linear algebra.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, log1p, lgamma, isfinite, fabs, sqrt
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_blas cimport dgemm, dgemv
from scipy.linalg.cython_lapack cimport dgetrf, dgetri
import scipy.optimize._lbfgsb as _lbfgsb

cnp.import_array()

cdef double HUGE_NLL = 1.0e100
cdef double MACHINE_EPS = np.finfo(np.float64).eps
cdef bint NEW_LBFGSB_API = "ln_task" in ((_lbfgsb.setulb.__doc__) or "")


cdef inline double _norm2(double* x, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double total = 0.0
    for i in range(n):
        total += x[i] * x[i]
    return sqrt(total)


cdef class TEvaluator:
    cdef cnp.ndarray x_arr
    cdef double* x_ptr
    cdef Py_ssize_t n
    cdef double df, logconst, cc

    def __cinit__(self, object x, double df):
        self.x_arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        self.x_ptr = <double*> cnp.PyArray_DATA(self.x_arr)
        self.n = self.x_arr.shape[0]
        self.df = df
        self.cc = df + 1.0
        self.logconst = (
            lgamma((df + 1.0) / 2.0)
            - lgamma(df / 2.0)
            - 0.5 * log(df * 3.1415926535897932384626433832795)
        )

    cdef double evaluate(self, double* par, double* grad) noexcept nogil:
        cdef Py_ssize_t i
        cdef double mu = par[0]
        cdef double eta = par[1]
        cdef double sigma = exp(eta)
        cdef double y, den
        cdef double loglik = 0.0
        cdef double gmu = 0.0
        cdef double geta = 0.0
        if (not isfinite(mu)) or (not isfinite(eta)) or (not isfinite(sigma)) or sigma <= 0.0:
            grad[0] = 0.0
            grad[1] = 0.0
            return HUGE_NLL
        for i in range(self.n):
            y = (self.x_ptr[i] - mu) / sigma
            den = self.df + y * y
            loglik += self.logconst - eta - 0.5 * self.cc * log1p((y * y) / self.df)
            gmu += -self.cc * y / (sigma * den)
            geta += 1.0 - self.cc * y * y / den
        grad[0] = gmu
        grad[1] = geta
        if not isfinite(loglik):
            grad[0] = 0.0
            grad[1] = 0.0
            return HUGE_NLL
        return -loglik


cdef class ICAEvaluator:
    cdef cnp.ndarray block_arr
    cdef double* block_ptr
    cdef int n, q, dim, lwork
    cdef double df, logconst, cc
    cdef double* centered
    cdef double* W
    cdef double* Winv
    cdef double* sources
    cdef double* scores
    cdef double* gradW
    cdef double* sumscore
    cdef double* tmpmu
    cdef double* work
    cdef int* ipiv

    def __cinit__(self, object block, double df):
        self.block_arr = np.ascontiguousarray(block, dtype=np.float64)
        if self.block_arr.ndim != 2:
            raise ValueError("block must be a two-dimensional array")
        self.block_ptr = <double*> cnp.PyArray_DATA(self.block_arr)
        self.n = <int> self.block_arr.shape[0]
        self.q = <int> self.block_arr.shape[1]
        self.dim = self.q + self.q * self.q
        self.df = df
        self.cc = df + 1.0
        self.logconst = (
            lgamma((df + 1.0) / 2.0)
            - lgamma(df / 2.0)
            - 0.5 * log(df * 3.1415926535897932384626433832795)
        )
        self.centered = <double*> malloc(max(1, self.n * self.q) * sizeof(double))
        self.W = <double*> malloc(max(1, self.q * self.q) * sizeof(double))
        self.Winv = <double*> malloc(max(1, self.q * self.q) * sizeof(double))
        self.sources = <double*> malloc(max(1, self.n * self.q) * sizeof(double))
        self.scores = <double*> malloc(max(1, self.n * self.q) * sizeof(double))
        self.gradW = <double*> malloc(max(1, self.q * self.q) * sizeof(double))
        self.sumscore = <double*> malloc(max(1, self.q) * sizeof(double))
        self.tmpmu = <double*> malloc(max(1, self.q) * sizeof(double))
        self.ipiv = <int*> malloc(max(1, self.q) * sizeof(int))
        self.lwork = max(1, self.q * 64)
        self.work = <double*> malloc(self.lwork * sizeof(double))
        if (
            self.centered == NULL or self.W == NULL or self.Winv == NULL
            or self.sources == NULL or self.scores == NULL or self.gradW == NULL
            or self.sumscore == NULL or self.tmpmu == NULL or self.ipiv == NULL
            or self.work == NULL
        ):
            raise MemoryError()

    def __dealloc__(self):
        if self.centered != NULL: free(self.centered)
        if self.W != NULL: free(self.W)
        if self.Winv != NULL: free(self.Winv)
        if self.sources != NULL: free(self.sources)
        if self.scores != NULL: free(self.scores)
        if self.gradW != NULL: free(self.gradW)
        if self.sumscore != NULL: free(self.sumscore)
        if self.tmpmu != NULL: free(self.tmpmu)
        if self.ipiv != NULL: free(self.ipiv)
        if self.work != NULL: free(self.work)

    cdef double evaluate(self, double* par, double* grad) noexcept nogil:
        cdef int i, j, a, info, inc = 1
        cdef double alpha = 1.0
        cdef double beta = 0.0
        cdef char N = <char> 78
        cdef char T = <char> 84
        cdef double logabsdet = 0.0
        cdef double loglik, value, score_value

        if self.q == 0:
            return 0.0

        # Column-major scratch arrays for BLAS/LAPACK.
        for j in range(self.q):
            for i in range(self.n):
                self.centered[i + self.n * j] = self.block_ptr[i * self.q + j] - par[j]
            for a in range(self.q):
                self.W[a + self.q * j] = par[self.q + a * self.q + j]
                self.Winv[a + self.q * j] = self.W[a + self.q * j]

        dgetrf(&self.q, &self.q, self.Winv, &self.q, self.ipiv, &info)
        if info != 0:
            for i in range(self.dim): grad[i] = 0.0
            return HUGE_NLL

        for i in range(self.q):
            value = self.Winv[i + self.q * i]
            if value == 0.0 or not isfinite(value):
                for j in range(self.dim): grad[j] = 0.0
                return HUGE_NLL
            logabsdet += log(fabs(value))

        dgetri(&self.q, self.Winv, &self.q, self.ipiv, self.work, &self.lwork, &info)
        if info != 0:
            for i in range(self.dim): grad[i] = 0.0
            return HUGE_NLL

        dgemm(&N, &T, &self.n, &self.q, &self.q, &alpha,
              self.centered, &self.n, self.W, &self.q, &beta,
              self.sources, &self.n)

        loglik = self.n * logabsdet
        for a in range(self.q):
            self.sumscore[a] = 0.0
            for i in range(self.n):
                value = self.sources[i + self.n * a]
                loglik += self.logconst - 0.5 * self.cc * log1p((value * value) / self.df)
                score_value = -self.cc * value / (self.df + value * value)
                self.scores[i + self.n * a] = score_value
                self.sumscore[a] += score_value

        dgemm(&T, &N, &self.q, &self.q, &self.n, &alpha,
              self.scores, &self.n, self.centered, &self.n, &beta,
              self.gradW, &self.q)
        dgemv(&T, &self.q, &self.q, &alpha, self.W, &self.q,
              self.sumscore, &inc, &beta, self.tmpmu, &inc)

        for j in range(self.q):
            grad[j] = self.tmpmu[j]
        for a in range(self.q):
            for j in range(self.q):
                grad[self.q + a * self.q + j] = -(
                    self.n * self.Winv[j + self.q * a]
                    + self.gradW[a + self.q * j]
                )

        if not isfinite(loglik):
            for i in range(self.dim): grad[i] = 0.0
            return HUGE_NLL
        return -loglik


cdef tuple _lbfgsb_t_new(
    TEvaluator evaluator,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] start,
    double lower_eta,
    int maxiter,
    double ftol,
    double gtol,
    int maxls,
    int maxcor,
):
    cdef int n = 2
    cdef int nit = 0
    cdef int nfev = 0
    cdef bint success = False
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = start.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lower = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] upper = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] nbd = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] g = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] wa = np.zeros(2*maxcor*n + 5*n + 11*maxcor*maxcor + 8*maxcor, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] iwa = np.zeros(3*n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] task = np.zeros(2, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] ln_task = np.zeros(2, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] lsave = np.zeros(4, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] isave = np.zeros(44, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] dsave = np.zeros(29, dtype=np.float64)
    cdef object f = np.array(0.0, dtype=np.float64)
    cdef double factr = ftol / MACHINE_EPS
    cdef double f_value

    nbd[1] = 1
    lower[1] = lower_eta
    if x[1] < lower_eta: x[1] = lower_eta

    while True:
        _lbfgsb.setulb(maxcor, x, lower, upper, nbd, f, g, factr, gtol,
                       wa, iwa, task, lsave, isave, dsave, maxls, ln_task)
        if task[0] == 3:
            with nogil:
                f_value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
            f[...] = f_value
            nfev += 1
        elif task[0] == 1:
            nit += 1
            if nit >= maxiter:
                break
        else:
            success = task[0] == 4
            break
    with nogil:
        f_value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
    return x, float(f_value), float(_norm2(<double*> cnp.PyArray_DATA(g), n)), bool(success), nit, nfev


cdef tuple _lbfgsb_ica_new(
    ICAEvaluator evaluator,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] start,
    int maxiter,
    double ftol,
    double gtol,
    int maxls,
    int maxcor,
):
    cdef int n = start.shape[0]
    cdef int nit = 0
    cdef int nfev = 0
    cdef bint success = False
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = start.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lower = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] upper = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] nbd = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] g = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] wa = np.zeros(2*maxcor*n + 5*n + 11*maxcor*maxcor + 8*maxcor, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] iwa = np.zeros(3*n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] task = np.zeros(2, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] ln_task = np.zeros(2, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] lsave = np.zeros(4, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] isave = np.zeros(44, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] dsave = np.zeros(29, dtype=np.float64)
    cdef object f = np.array(0.0, dtype=np.float64)
    cdef double factr = ftol / MACHINE_EPS
    cdef double f_value

    while True:
        _lbfgsb.setulb(maxcor, x, lower, upper, nbd, f, g, factr, gtol,
                       wa, iwa, task, lsave, isave, dsave, maxls, ln_task)
        if task[0] == 3:
            with nogil:
                f_value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
            f[...] = f_value
            nfev += 1
        elif task[0] == 1:
            nit += 1
            if nit >= maxiter:
                break
        else:
            success = task[0] == 4
            break
    with nogil:
        f_value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
    return x, float(f_value), float(_norm2(<double*> cnp.PyArray_DATA(g), n)), bool(success), nit, nfev


def _old_task_text(task):
    return bytes(task[0]).split(b"\x00", 1)[0]


def _lbfgsb_t_old(TEvaluator evaluator, start, lower_eta, maxiter, ftol, gtol, maxls, maxcor):
    # Compatibility path for SciPy versions with the legacy character-task API.
    x = np.ascontiguousarray(start, dtype=np.float64).copy()
    n = x.size
    lower = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    nbd = np.zeros(n, dtype=np.int32)
    lower[1] = lower_eta
    nbd[1] = 1
    x[1] = max(x[1], lower_eta)
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    wa = np.zeros(2*maxcor*n + 5*n + 11*maxcor*maxcor + 8*maxcor, dtype=np.float64)
    iwa = np.zeros(3*n, dtype=np.int32)
    task = np.zeros(1, dtype="S60"); task[:] = b"START"
    csave = np.zeros(1, dtype="S60")
    lsave = np.zeros(4, dtype=np.int32)
    isave = np.zeros(44, dtype=np.int32)
    dsave = np.zeros(29, dtype=np.float64)
    factr = ftol / np.finfo(float).eps
    nit = 0; nfev = 0; success = False
    while True:
        _lbfgsb.setulb(maxcor, x, lower, upper, nbd, f, g, factr, gtol,
                       wa, iwa, task, -1, csave, lsave, isave, dsave, maxls)
        text = _old_task_text(task)
        if text.startswith(b"FG"):
            value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
            f[...] = value; nfev += 1
        elif text.startswith(b"NEW_X"):
            nit += 1
            if nit >= maxiter: break
        else:
            success = text.startswith(b"CONV")
            break
    value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
    return x, float(value), float(np.linalg.norm(g)), bool(success), nit, nfev


def _lbfgsb_ica_old(ICAEvaluator evaluator, start, maxiter, ftol, gtol, maxls, maxcor):
    x = np.ascontiguousarray(start, dtype=np.float64).copy()
    n = x.size
    lower = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    nbd = np.zeros(n, dtype=np.int32)
    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    wa = np.zeros(2*maxcor*n + 5*n + 11*maxcor*maxcor + 8*maxcor, dtype=np.float64)
    iwa = np.zeros(3*n, dtype=np.int32)
    task = np.zeros(1, dtype="S60"); task[:] = b"START"
    csave = np.zeros(1, dtype="S60")
    lsave = np.zeros(4, dtype=np.int32)
    isave = np.zeros(44, dtype=np.int32)
    dsave = np.zeros(29, dtype=np.float64)
    factr = ftol / np.finfo(float).eps
    nit = 0; nfev = 0; success = False
    while True:
        _lbfgsb.setulb(maxcor, x, lower, upper, nbd, f, g, factr, gtol,
                       wa, iwa, task, -1, csave, lsave, isave, dsave, maxls)
        text = _old_task_text(task)
        if text.startswith(b"FG"):
            value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
            f[...] = value; nfev += 1
        elif text.startswith(b"NEW_X"):
            nit += 1
            if nit >= maxiter: break
        else:
            success = text.startswith(b"CONV")
            break
    value = evaluator.evaluate(<double*> cnp.PyArray_DATA(x), <double*> cnp.PyArray_DATA(g))
    return x, float(value), float(np.linalg.norm(g)), bool(success), nit, nfev


def optimize_t(
    object x,
    double df,
    object start,
    double lower_eta,
    int maxiter=250,
    double ftol=1.0e-12,
    double gtol=1.0e-8,
    int maxls=40,
    int maxcor=10,
):
    cdef TEvaluator evaluator = TEvaluator(x, df)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] start_arr = np.ascontiguousarray(start, dtype=np.float64).reshape(-1)
    if start_arr.shape[0] != 2:
        raise ValueError("Student-t optimizer requires two starting parameters")
    if NEW_LBFGSB_API:
        return _lbfgsb_t_new(evaluator, start_arr, lower_eta, maxiter, ftol, gtol, maxls, maxcor)
    return _lbfgsb_t_old(evaluator, start_arr, lower_eta, maxiter, ftol, gtol, maxls, maxcor)


def optimize_ica_multistart(
    object block,
    double df,
    object starts,
    int maxiter=500,
    double ftol=1.0e-10,
    double gtol=1.0e-6,
    int maxls=50,
    int maxcor=10,
):
    cdef ICAEvaluator evaluator = ICAEvaluator(block, df)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] starts_arr = np.ascontiguousarray(starts, dtype=np.float64)
    cdef int nstarts = starts_arr.shape[0]
    cdef int dim = starts_arr.shape[1]
    cdef int s, best_start = -1
    cdef double start_nll, final_nll, grad_norm
    cdef bint success
    cdef int nit, nfev, total_nit = 0, total_nfev = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] start
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] candidate
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tmp_grad = np.empty(dim, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] best_x = np.empty(dim, dtype=np.float64)
    cdef double best_nll = HUGE_NLL
    cdef double best_grad_norm = HUGE_NLL
    cdef bint best_success = False
    cdef tuple result

    if nstarts <= 0:
        raise ValueError("At least one affine-ICA start is required")
    if dim != evaluator.dim:
        raise ValueError(f"Each start must have length {evaluator.dim}; got {dim}")

    for s in range(nstarts):
        start = np.ascontiguousarray(starts_arr[s], dtype=np.float64)
        with nogil:
            start_nll = evaluator.evaluate(<double*> cnp.PyArray_DATA(start), <double*> cnp.PyArray_DATA(tmp_grad))
        if NEW_LBFGSB_API:
            result = _lbfgsb_ica_new(evaluator, start, maxiter, ftol, gtol, maxls, maxcor)
        else:
            result = _lbfgsb_ica_old(evaluator, start, maxiter, ftol, gtol, maxls, maxcor)
        candidate = np.ascontiguousarray(result[0], dtype=np.float64)
        final_nll = float(result[1])
        grad_norm = float(result[2])
        success = bool(result[3])
        nit = int(result[4]); nfev = int(result[5])
        total_nit += nit; total_nfev += nfev

        # Preserve the exact safety rule used by the verified Python core.
        if (not np.isfinite(final_nll)) or final_nll > start_nll:
            candidate = start
            final_nll = start_nll
            with nogil:
                final_nll = evaluator.evaluate(<double*> cnp.PyArray_DATA(candidate), <double*> cnp.PyArray_DATA(tmp_grad))
                grad_norm = _norm2(<double*> cnp.PyArray_DATA(tmp_grad), dim)
            success = False

        if final_nll < best_nll:
            best_nll = final_nll
            best_x[:] = candidate
            best_grad_norm = grad_norm
            best_success = success
            best_start = s

    if best_start < 0 or not np.isfinite(best_nll):
        raise RuntimeError("No finite affine-ICA profile likelihood was found")

    return {
        "x": np.asarray(best_x),
        "nll": float(best_nll),
        "gradient_norm": float(best_grad_norm),
        "success": bool(best_success),
        "best_start": int(best_start),
        "total_iterations": int(total_nit),
        "total_function_evaluations": int(total_nfev),
        "lbfgsb_api": "integer-task" if NEW_LBFGSB_API else "legacy-character-task",
    }


def t_nll_gradient(object parameters, object x, double df):
    cdef TEvaluator evaluator = TEvaluator(x, df)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] par = np.ascontiguousarray(parameters, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] grad = np.empty(2, dtype=np.float64)
    cdef double value
    if par.shape[0] != 2:
        raise ValueError("Expected two parameters")
    with nogil:
        value = evaluator.evaluate(<double*> cnp.PyArray_DATA(par), <double*> cnp.PyArray_DATA(grad))
    return float(value), grad


def t_observed_hessian(double mu, double eta, object x, double df):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x_arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t n = x_arr.shape[0]
    cdef double* xp = <double*> cnp.PyArray_DATA(x_arr)
    cdef Py_ssize_t i
    cdef double sigma = exp(eta), y, den, cc = df + 1.0
    cdef double h00 = 0.0, h01 = 0.0, h11 = 0.0
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((2, 2), dtype=np.float64)
    with nogil:
        for i in range(n):
            y = (xp[i] - mu) / sigma
            den = df + y * y
            h00 += cc * (df - y * y) / (sigma * sigma * den * den)
            h01 += 2.0 * cc * df * y / (sigma * den * den)
            h11 += 2.0 * cc * df * y * y / (den * den)
    out[0, 0] = h00; out[0, 1] = h01
    out[1, 0] = h01; out[1, 1] = h11
    return out


def ica_nll_gradient(object parameters, object block, double df):
    cdef ICAEvaluator evaluator = ICAEvaluator(block, df)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] par = np.ascontiguousarray(parameters, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] grad = np.empty(evaluator.dim, dtype=np.float64)
    cdef double value
    if par.shape[0] != evaluator.dim:
        raise ValueError(f"Expected {evaluator.dim} parameters")
    with nogil:
        value = evaluator.evaluate(<double*> cnp.PyArray_DATA(par), <double*> cnp.PyArray_DATA(grad))
    return float(value), grad


def backend_info():
    return {
        "name": "full_optimizer_backend_v4",
        "implementation": "Cython/C + SciPy compiled L-BFGS-B + BLAS/LAPACK",
        "version": "1.0.0",
        "full_optimizer_loop_compiled": True,
        "multistart_selection_compiled": True,
        "lbfgsb_api": "integer-task" if NEW_LBFGSB_API else "legacy-character-task",
    }
