cimport numpy as np
import cython
from cython import boundscheck, wraparound
import time
    
cdef extern from "Gather.h":
    cdef cppclass FastIV:
        FastIV(int, int, int, int ,int, int*, int*) except +
        int R, B, mf, topk, node
        int block
        int* inv_lookup
        int* counts
        
        # void createIndex(int, long*, int)
        void FC(int*, int, int*, int*)

@boundscheck(False)
@wraparound(False)
cdef class PyFastIV:
    cdef FastIV *thisptr
    def __cinit__(self, int R, int block, int B, int mf, int topk, np.ndarray[int, ndim=1,  mode="c"] inv_lookup, np.ndarray[int, ndim=1,  mode="c"] counts ):
        self.thisptr = new FastIV( R, block, B, mf, topk, &inv_lookup[0], &counts[0])
    # def createIndex(self, int i, np.ndarray[long, ndim=1,  mode="c"] list, int L):
    #     self.thisptr.createIndex(i, &list[0], L)
    def FC(self, np.ndarray[int, ndim=1,  mode="c"] top_buckets_, int maxsize, np.ndarray[int, ndim=1,  mode="c"] candidates, np.ndarray[int, ndim=1,  mode="c"] candSize):
        return self.thisptr.FC(&top_buckets_[0], maxsize,  &candidates[0], &candSize[0])

