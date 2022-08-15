#include <iomanip>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <math.h>
#include <sstream>
#include <string>
#include <string.h>
#include <algorithm>
#include "Gather.h"
#include <iostream>
#include <unordered_map>
#include <typeinfo>


using namespace std;

// k is the nnz, m is the output size, N is the counter size
FastIV::FastIV(int R, int block, int B, int mf, int topk, int* inv_lookup, int* counts){
    R_ =R; //k
    block_ = block; //N
    B_ = B; //m
    threshold = mf; //r, threshold
    topk_ = topk;
    inv_lookup_ = inv_lookup;
    counts_ = counts;
    tempe = 0;
    std::cout << "params: "<<  R_<<" "<< block_<<" "<<  B_<<" "<<  threshold<<" "<< '\n';

    counter = new uint8_t[block_]; //can go further down with each unit= log(maxCount)
    // memset(counter, 0, sizeof(counter));  //tales almost same time 
    #pragma omp parallel for num_threads(32) // or #pragma omp simd
    for(long i = 0; i < block_; i++){  
        counter[i] = 0;
    }
    // InvIndex = new vector<int>[m_]; //array of vectors
    // long* inv_lookup = new long[block*R*Parts]; 
}


// ///////////////////////////////////////////vFor IRLI////////////////////////////////////////////////////////////////////
// candSize = self.fastIv.FC(top_buckets_[i,:,1,:], 60000, candidates)

void FastIV::FC(int* top_buckets_, int maxsize, int* candidates, int* candSize){  
    #pragma omp parallel for num_threads(32)
    for(long i = maxsize; i >=0 ; --i){
        candidates[i] =0;
    }
    // #pragma omp parallel for num_threads(4)
    // chrono::time_point<chrono::high_resolution_clock> t0 = chrono::high_resolution_clock::now();

    // uint8_t* counter = new uint8_t[block_]; //can go further down with each unit= log(maxCount)
    // // memset(counter, 0, sizeof(counter));  //tales almost same time 
    // #pragma omp parallel for num_threads(32) // or #pragma omp simd
    // for(long i = 0; i < block_; i++){  // MAJOR TIME TAKING PROCESS----TIP, DONT INIT ENTIRE COUNTER FOR A NEW QUERY, USE FRON PREVIIOUS 0'ed at places used
    //     counter[i] = 0;
    // }

    candSize[0] = 0;
    
    // long long int bucket = 0;
    int bucketsz = 0;
    int stpt = 0;
    int r = 0;
    // int pos,pos2 = 0;
    long cnt =0;
    if (R_==1){
        for(int k =0; k<topk_; k++) {
            stpt  = counts_[(r)*B_+ top_buckets_[topk_*r+ k]] +(r)*block_;
            bucketsz = counts_[(r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(r)*B_+ top_buckets_[topk_*r+ k]];
            for (int i=0; i<bucketsz; i++){ 
                if (cnt<maxsize){
                candidates[cnt] = inv_lookup_[stpt+i];
                cnt++;}
            }
        }
        candSize[0] = cnt;
    }

    if (R_>1){
    // chrono::time_point<chrono::high_resolution_clock> t1 = chrono::high_resolution_clock::now();
    // cout << "d1: "<<chrono::duration_cast<chrono::nanoseconds>(t1-t0).count()/1000000.0 << "msec\n";
    // #pragma omp parallel for num_threads(topk_) // this one doesn't create issues
    // cout << "h1 "<<endl;
        // #pragma omp parallel for num_threads(8)
    r = 0;
    for(int k =0; k<topk_; k++) {
        //////simpler ///////////
        // bucket = top_buckets_[topk_*r+ k];
        // pos = (part*R_+ r)*B_;
        // pos2 = (part*R_+ r)*block_;
        // stpt  = counts_[pos+ bucket];
        // bucketsz = counts_[pos + bucket+1] - stpt;
        // stpt = stpt+pos2;
        /////////////////////////
        
        stpt  = counts_[(r)*B_+ top_buckets_[topk_*r+ k]] +(r)*block_;
        bucketsz = counts_[(r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(r)*B_+ top_buckets_[topk_*r+ k]];
        // #pragma omp simd
        // cout << "h1_2 "<<endl;
        for (int i=0; i<bucketsz; i++){ // just increment in first pass
            // cout<<part<<" "<<stpt+i<<" "<<inv_lookup_[stpt+i]<<endl;
            counter[inv_lookup_[stpt+i]]+=1;
        }
    }
    // chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
    // cout << "d2: "<<chrono::duration_cast<chrono::nanoseconds>(t2-t1).count()/1000000.0 << "msec\n";
    long cnt =0;
    for(int r=1; r<R_; r++){
        for(int k =0; k<topk_; k++) {
            stpt  = counts_[(r)*B_+ top_buckets_[topk_*r+ k]] +(r)*block_;
            bucketsz = counts_[(r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(r)*B_+ top_buckets_[topk_*r+ k]];
            for (int i=0; i<bucketsz; i++){
                if(counter[inv_lookup_[stpt+i]]<threshold){
                    counter[inv_lookup_[stpt+i]]+=1;
                }
                if(counter[inv_lookup_[stpt+i]]==threshold&& cnt < maxsize){
                    candidates[cnt] = inv_lookup_[stpt+i];
                    // candidates[cnt] = inv_lookup_[stpt+i]+ (part*block_);
                    cnt++;
                    counter[inv_lookup_[stpt+i]]+=1;
                }
            }
        }
    }
    candSize[0] = cnt;
    //cleanup
    // chrono::time_point<chrono::high_resolution_clock> t3 = chrono::high_resolution_clock::now();
    // cout << "d3: "<<chrono::duration_cast<chrono::nanoseconds>(t3-t2).count()/1000000.0 << "msec\n";
    
    // if (part<Parts_-1){
        // #pragma omp parallel for num_threads(8) collapse(2)
        for(int r=0; r<R_; r++){
            for(int k =0; k<topk_; k++) {
                stpt  = counts_[(r)*B_+ top_buckets_[topk_*r+ k]] +(r)*block_;
                bucketsz = counts_[(r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(r)*B_+ top_buckets_[topk_*r+ k]];
                for (int i=0; i<bucketsz; i++){ // just increment in first pass
                    counter[inv_lookup_[stpt+i]]=0;
                }
            }
        }
    // }    
    // chrono::time_point<chrono::high_resolution_clock> t4 = chrono::high_resolution_clock::now();
    // cout << "d4: "<<chrono::duration_cast<chrono::nanoseconds>(t4-t3).count()/1000000.0 << "msec\n";
    // if (part == Parts_-1) {
    //     break;}
    }
    
    // delete [] counter;
    // return cnts;
}