#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <set>
#include <iterator>
#include <stdlib.h>    
#include <numeric>
#include <algorithm>
#include <string> 
#include <cstdint>
#include <map>

#include <cstdio>
#include <bits/stdc++.h>
#include <sys/stat.h>

#include <iomanip>
#include <cmath>
#include <stdio.h>

#ifdef __AVX__
  #include <immintrin.h>
#else
  #warning AVX is not available. Code will not compile!
#endif
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif


//include something for map
using namespace std;

double computeRecall(vector<vector<int>> answer, vector<vector<int>> guess);
vector<vector<int>> computeGroundTruth(vector<vector<int>> queryset, vector<set<string>> queryprops, vector<vector<int>> data, vector<set<string>> properties, int num_results);
vector<uint32_t> argTopK(float* query, float* vectors, uint32_t d, uint32_t N, vector<uint32_t> idx, uint32_t idxSize, uint32_t k, vector<float> topkDist);
float L2sim(float* a, float* b, float norm_bsq, size_t d);
float L2SqrSIMD16ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty);
float IP(float* a, float* b, size_t d);
float IPSIMD16ExtAVX(float *pVect1, float *pVect2, size_t qty);
uint16_t getclusterPart(uint16_t* maxMC, vector<uint16_t> &props, int treelen);
bool not_in(uint16_t x, uint16_t* a, int h);

double RecallAtK(int* answer, int* guess, size_t k, size_t nq);
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out);
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out);

void randomShuffle(int* v , int l, int u);



