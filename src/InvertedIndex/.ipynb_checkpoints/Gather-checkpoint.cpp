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

// k is the nnz, m is the output size
FastIV::FastIV(int k, int m, long N){
    k_ = k;
    N_ = N;
    m_ = m;
    std::cout << "output dim: " <<m<< '\n';
    InvIndex = new vector<int>[m_]; //array of vectors
    std::cout <<k_<< '\n';
    std::cout <<N_<< '\n';
    std::cout <<m_<< '\n';
}
// L is bucket size
void FastIV::createIndex(int i, long* list, int L){
    for(int l=0; l<L; l++){
        InvIndex[i].push_back(list[l]);
    }
//     std::cout <<InvIndex[i].size()<< '\n';
}


// ///////////////////////////////////////////version 1////////////////////////////////////////////////////////////////////

// long FastIV::FC(long* hashIndex, int threshold, int maxsize, long* candidates){
// //     std::cout <<k_<< '\n';
// //     short int counter[N_] = {0}; //init with all 0s
    
//     short int* counter = new short int[N_];
//     chrono::time_point<chrono::high_resolution_clock> t0 = chrono::high_resolution_clock::now();
//     for(long i = 0; i < N_; i++){
//         counter[i] = 0;
//     }
// //     chrono::time_point<chrono::high_resolution_clock> t1 = chrono::high_resolution_clock::now();
// //     cout << "d0: "<<chrono::duration_cast<chrono::nanoseconds>(t1-t0).count()/1000000.0 << "msec\n";
    
// //     std::cout <<N_<< '\n';
// //     std::cout <<m_<< '\n';
//     for(int h=0; h<k_; h++){
//         for (int i=0; i<InvIndex[hashIndex[h]].size(); i++){
//             counter[InvIndex[hashIndex[h]][i]]+=1;
//         }
//     }
// //     chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
// //     cout << "d1: "<<chrono::duration_cast<chrono::nanoseconds>(t2-t1).count()/1000000.0 << "msec\n";
    
// //     std::cout <<"here1"<< '\n';
//     long cnt =0;
//     for(long i = maxsize; i >=0 ; --i){
//         candidates[i] =0;
//     }
// //     chrono::time_point<chrono::high_resolution_clock> t3 = chrono::high_resolution_clock::now();
// //     cout << "d2: "<<chrono::duration_cast<chrono::nanoseconds>(t3-t2).count()/1000000.0 << "msec\n";
    
// //     std::cout <<"here2"<< '\n';
//     for(long i = 0; i < N_; i++){
//         if (counter[i]>=threshold && cnt <=maxsize){
//             candidates[cnt] = i;
// //             std::cout << "here3 " <<candidates[cnt]<< '\n';
//             cnt++;
//         }
//     }
//     delete [] counter;
// //     chrono::time_point<chrono::high_resolution_clock> t4 = chrono::high_resolution_clock::now();
// //     cout << "d3: "<<chrono::duration_cast<chrono::nanoseconds>(t4-t3).count()/1000000.0 << "msec\n";
// //     std::cout << cnt<< '\n';
//     return cnt;
// }

// ///////////////////////////////////////////version 3////////////////////////////////////////////////////////////////////

long FastIV::FC(long* hashIndex, int threshold, int maxsize, long* candidates){  
    
//     chrono::time_point<chrono::high_resolution_clock> t0 = chrono::high_resolution_clock::now();
    
    short int* counter = new short int[N_];
    for(long i = 0; i < N_; i++){
        counter[i] = 0;
    }

//     chrono::time_point<chrono::high_resolution_clock> t1 = chrono::high_resolution_clock::now();
//     cout << "d0: "<<chrono::duration_cast<chrono::nanoseconds>(t1-t0).count()/1000000.0 << "msec\n";

    long cnt =0;
    for(long i = maxsize; i >=0 ; --i){
        candidates[i] =0;
    }
    
//     chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
//     cout << "d1: "<<chrono::duration_cast<chrono::nanoseconds>(t2-t1).count()/1000000.0 << "msec\n";
    
    for(int h=0; h<k_; h++){
        for (int i=0; i<InvIndex[hashIndex[h]].size(); i++){
            if (counter[InvIndex[hashIndex[h]][i]]<threshold){
                counter[InvIndex[hashIndex[h]][i]]+=1;
            }
            if (counter[InvIndex[hashIndex[h]][i]]==threshold && cnt < maxsize){
                candidates[cnt] = InvIndex[hashIndex[h]][i];
                cnt++;
                counter[InvIndex[hashIndex[h]][i]]+=1;
            }
        }
    }

//     chrono::time_point<chrono::high_resolution_clock> t3 = chrono::high_resolution_clock::now();
//     cout << "d2: "<<chrono::duration_cast<chrono::nanoseconds>(t3-t2).count()/1000000.0 << "msec\n";

    delete [] counter;
    return cnt;
}


// ///////////////////////////////////////////version 2////////////////////////////////////////////////////////////////////

// void FastIV::FC(long* hashIndex, int threshold, int maxsize, long* candidates){
// //     short int* counter = new short int[N_];
//     std::unordered_map<long, short int> counter;
//     counter.reserve(10000);
    
//     chrono::time_point<chrono::high_resolution_clock> t1 = chrono::high_resolution_clock::now();
//     for(int h=0; h<k_; h++){
//         for (int i=0; i<InvIndex[hashIndex[h]].size(); i++){
//             if (counter.find(InvIndex[hashIndex[h]][i]) == counter.end()){
//                 counter[InvIndex[hashIndex[h]][i]]=0;
//             }
//             counter[InvIndex[hashIndex[h]][i]]+=1;
//         }
//     }
//     chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
//     cout << "d1: "<<chrono::duration_cast<chrono::nanoseconds>(t2-t1).count()/1000000.0 << "msec\n";
    
//     long cnt =0;
//     for(long i = maxsize; i >=0 ; --i){
//         candidates[i] =0;
//     }
//     chrono::time_point<chrono::high_resolution_clock> t3 = chrono::high_resolution_clock::now();
//     cout << "d2: "<<chrono::duration_cast<chrono::nanoseconds>(t3-t2).count()/1000000.0 << "msec\n";
    
//     for (auto const& x : counter){
// //     for (auto x : counter)
// //         cout << x.first << " " << x.second << endl;
//         if (x.second>=threshold && cnt <=maxsize){
//             candidates[cnt] = x.first;
//             cnt++;
//         }
//     }
//     chrono::time_point<chrono::high_resolution_clock> t4 = chrono::high_resolution_clock::now();
//     cout << "d3: "<<chrono::duration_cast<chrono::nanoseconds>(t4-t3).count()/1000000.0 << "msec\n";
// //     std::cout << cnt<< ' ';
// }



float FastIV::distComp(float* query, float* train_data, int d, long* candidates, int cansz, float* ip, int n_threads)
{
    int cand = 0;
        
    float* final_cand = new float[cansz];
    float* temp = new float[d];
    float assign=0;
    float mm=0;
//     # pragma omp parallel num_threads ( n_threads )
    chrono::time_point<chrono::high_resolution_clock> t0 = chrono::high_resolution_clock::now();

    for (int i=0; i<cansz; i++){
        cand = candidates[i];
        std::copy(train_data+ (cand*d),train_data+ ((cand+1)*d), temp);
//         for (int j=0; j<d; j++){
// //             temp = train_data[cand*d+j];
//             std::copy(train_data+ (cand*d),train_data+ ((cand+1)*d), temp);
// //             final_cand[i] +=temp*query[j];
// //             chrono::time_point<chrono::high_resolution_clock> t2 = chrono::high_resolution_clock::now();
// //             ip[i] += temp*query[j];
//         } 
    }
    chrono::time_point<chrono::high_resolution_clock> t1 = chrono::high_resolution_clock::now();
    cout<<chrono::duration_cast<chrono::nanoseconds>(t1-t0).count()/1000000.0<<endl;
//     cout << "assign: "<<assign << "msec\n";
//     cout << "mm: "<<mm << "msec\n";
//     cout<<final_cand[0];
}
