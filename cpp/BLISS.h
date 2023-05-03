// function to bins
#pragma once
#include <iostream>
#include <fstream>
#include <stdlib.h>    
#include <numeric>
#include <algorithm>
#include <string> 
#include <vector>
#include <map>
#include <random>
#include <sys/time.h>
#include <sys/stat.h>
#include <chrono>
#include "utils.h"

using namespace std;
class BLISS
{
    public:
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
        // BLISS* bliss;
        uint32_t d, nb, nc, k;

        int s1;
        int s2;
        int s3;
        float* weights;
        
        BLISS(float* data, size_t d_, size_t nb_, size_t nc_){
            dataset = data; // data
            d =d_; // dim
            nb = nb_; //num data points
            nc = nc_; // num clusters
            s1 = d;
            s2 = 256;
            s3 = nc;
            weights = new float[s2*(d+1) +nc*(s2+1)];
        }

        void getscore(float* input, float* last){
            float* hd = new float[s2];
            for (uint32_t id=0; id<s2; id++){
                hd[id] = IPSIMD16ExtAVX(input, weights+ id*(s1+1), s1);
                hd[id] += *(weights+ id*(s1+1) +s1); //bias
                if (hd[id]<0) hd[id] =0; // Relu 0 if negative
            }
            float* L1 = weights+ (s1+1)*s2;

            for (uint32_t id=0; id<s3; id++){
                last[id] = IPSIMD16ExtAVX(hd, L1+ id*(s2+1), s2); //multiply
                last[id] += *(L1+ id*(s2+1) +s2); //bias
            }
            // last = softmax(last);
            // return last;
        }

        uint32_t top(float* input){ 
            float* hd = new float[s2];
            for (uint32_t id=0; id<s2; id++){
                hd[id] = IPSIMD16ExtAVX(input, weights+ id*(s1+1), s1);
                hd[id] += *(weights+ id*(s1+1) +s1); //bias
                if (hd[id]<0) hd[id] =0; // Relu 0 if negative
            }
            // for(uint32_t i = 0; i < 20; ++i) {
            //     cout<<hd[i]<<" ";
            // }
            float* L1 = weights+ (s1+1)*s2;
            uint32_t bin;
            float maxscore = -1000000; 
            for (uint32_t id=0; id<s3; id++){
                float temp =0;
                temp = IPSIMD16ExtAVX(hd, L1+ id*(s2+1), s2); //multiply
                temp += *(L1+ id*(s2+1) +s2); //bias
                if (temp>maxscore) {
                maxscore=temp;
                bin = id;}
            }
            return bin;
        }

        void load(string modelpath){
            // load .dat
            FILE* f = fopen(modelpath.c_str(), "rb");
            fread(weights, sizeof(float), 256*129 +1024*257, f);
            fclose (f);
        } 
};


