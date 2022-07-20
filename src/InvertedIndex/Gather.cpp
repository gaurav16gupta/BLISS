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
FastIV::FastIV(int Parts, int R, int block, int B, int mf, int topk, int node, int* inv_lookup, int* counts){
    Parts_ = Parts; // these many times everything
    R_ =R; //k
    block_ = block; //N
    B_ = B; //m
    threshold = mf; //r, threshold
    topk_ = topk;
    node_ = node;
    inv_lookup_ = inv_lookup;
    counts_ = counts;
    tempe = 0;
    std::cout << "params: " << Parts_<<" "<<  R_<<" "<< block_<<" "<<  B_<<" "<<  threshold<<" "<<  node_<<" "<< '\n';

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

    for (int part=0; part<Parts_; part++){
        candSize[part] = 0;
        
        // long long int bucket = 0;
        int bucketsz = 0;
        int stpt = 0;
        int r = 0;
        // int pos,pos2 = 0;
        long cnt =part*maxsize/Parts_;
        if (R_==1){
            for(int k =0; k<topk_; k++) {
                stpt  = counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]] +(part*R_+ r)*block_;
                bucketsz = counts_[(part*R_+ r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]];
                for (int i=0; i<bucketsz; i++){ 
                    if (cnt<maxsize){
                    candidates[cnt] = inv_lookup_[stpt+i];
                    cnt++;}
                }
            }
            candSize[part] = cnt - part*maxsize/Parts_;
        }

        if (R_>1){
        
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
            
            stpt  = counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]] +(part*R_+ r)*block_;
            bucketsz = counts_[(part*R_+ r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]];
            // #pragma omp simd
            // cout << "h1_2 "<<endl;
            for (int i=0; i<bucketsz; i++){ // just increment in first pass
                // cout<<part<<" "<<stpt+i<<" "<<inv_lookup_[stpt+i]<<endl;
                counter[inv_lookup_[stpt+i]]+=1;
            }
        }
        // chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
        // cout << "d2: "<<chrono::duration_cast<chrono::nanoseconds>(t2-t1).count()/1000000.0 << "msec\n";
        long cnt =part*maxsize/Parts_;
        for(int r=1; r<R_; r++){
            for(int k =0; k<topk_; k++) {
                stpt  = counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]] +(part*R_+ r)*block_;
                bucketsz = counts_[(part*R_+ r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]];
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
        candSize[part] = cnt - part*maxsize/Parts_;
        //cleanup
        // chrono::time_point<chrono::high_resolution_clock> t3 = chrono::high_resolution_clock::now();
        // cout << "d3: "<<chrono::duration_cast<chrono::nanoseconds>(t3-t2).count()/1000000.0 << "msec\n";
        
        // if (part<Parts_-1){
            // #pragma omp parallel for num_threads(8) collapse(2)
            for(int r=0; r<R_; r++){
                for(int k =0; k<topk_; k++) {
                    stpt  = counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]] +(part*R_+ r)*block_;
                    bucketsz = counts_[(part*R_+ r)*B_ + top_buckets_[topk_*r+ k]+1] - counts_[(part*R_+ r)*B_+ top_buckets_[topk_*r+ k]];
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
    }
    // delete [] counter;
    // return cnts;
}