#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string> 
#include <map>
#include <random>
#include <sys/time.h>
#include <sys/stat.h>
#include <chrono>

#include "FaissKMeans.h"
#include "BLISS.h"
#include "utils.h"

using namespace std;

class NNsearch
{
    public:
        NNsearch(float* data, size_t d_, size_t nb_, size_t nc_);
        void make_index(string indexpath, string algo);
        void loadIndex(string indexpath);
        uint32_t query(float* queryset, int nq, int num_results, int nprobe);
        uint32_t findNearestNeighbor(float* query, int num_results, int nprobe, size_t qnum);

        float *dataset; //use <dtype> array instead
        float *dataset_reordered;
        float *centroids; 
        float* cen_norms;
        float* data_norms;
        float *data_norms_reordered;
        uint32_t* invLookup;
        uint32_t* Lookup; 
        uint32_t* counts;
        int32_t* neighbor_set;
        int numAttr;

        // Kmeans* kmeans;
        BLISS* bliss;
        uint32_t d, nb, nc, k;
};