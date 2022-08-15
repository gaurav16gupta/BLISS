import numpy as np
cimport numpy as np
import cython
from cython import boundscheck, wraparound
import time

# cdef extern from "Gather.h":
#     void cgather_batch(long*, long*, long*, long*, long*, double*, int, int, int, int, int) except +
#     # void cgather_K(float*, long*, float*, long*, int, int, int, int,int) except +

# @boundscheck(False)
# @wraparound(False)
# def scoreAgg( np.ndarray[long, ndim=2,  mode="c"] label_buckets, 
#             np.ndarray[long, ndim=2, mode="c"] counts, 
#             np.ndarray[long, ndim=2,  mode="c"] cumCounts,
#             np.ndarray[long, ndim=1,  mode="c"]candidates,
#             np.ndarray[long, ndim=2,  mode="c"] bestbins,  
#             np.ndarray[double, ndim=2,  mode="c"] bestbin_score, int R, int B, int N, int m, int maxsize): 
#     cgather_batch(&label_buckets[0,0], &counts[0,0], &cumCounts[0,0], &candidates[0], &bestbins[0,0], &bestbin_score[0,0], R, B, N, m, maxsize)

    
# new code
cdef extern from "Gather.h":
    cdef cppclass FastIV:
        FastIV(int, int, long) except +
        int k, m, N
        void createIndex(int, long*, int)
        long FC(long*, int, int, long* )
        float distComp(float*, float*, int, long*, int, float*, int)

@boundscheck(False)
@wraparound(False)
cdef class PyFastIV:
    cdef FastIV *thisptr
    def __cinit__(self, int k, int m, long N):
        self.thisptr = new FastIV(k, m, N)
    def createIndex(self, int i, np.ndarray[long, ndim=1,  mode="c"] list, int L):
        self.thisptr.createIndex(i, &list[0], L)
    def FC(self, np.ndarray[long, ndim=1,  mode="c"] hashIndex, int threshold, int maxsize, np.ndarray[long, ndim=1,  mode="c"] candidates):
        return self.thisptr.FC(&hashIndex[0], threshold, maxsize, &candidates[0])
    def distComp(self, np.ndarray[float, ndim=1,  mode="c"] query, np.ndarray[float, ndim=2,  mode="c"] train_data, int d, np.ndarray[long, ndim=1,  mode="c"] candidates, int cansz, np.ndarray[float, ndim=1,  mode="c"] ip, int n_threads ):
         return self.thisptr.distComp(& query[0], &train_data[0,0], d, &candidates[0], cansz, &ip[0], n_threads)




