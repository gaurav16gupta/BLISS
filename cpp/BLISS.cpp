#include "BLISS.h"
#include <omp.h>

using namespace std;

//NN index
void BLISS::make_index(string indexpath, string algo){
    data_norms = new float[nb]{0};
    Lookup= new uint32_t[nb];
    counts = new uint32_t[nc+1]{0};

    load(indexpath+"/model.bin");
    for(uint32_t j = 0; j < nb; ++j){  
        for(uint32_t k = 0; k < d; ++k) {    
            data_norms[j]=0;             
            data_norms[j] += dataset[j*d +k]*dataset[j*d +k];        
        } 
        data_norms[j]=data_norms[j]/2;
    }
    
    uint32_t* invLookup = new uint32_t[nb];
    //get best score cluster
    #pragma omp parallel for  
    for(uint32_t i = 0; i < nb; ++i) {  
        invLookup[i] = top(dataset+ i*d);   
    }
     for(uint32_t i = 0; i < nb; ++i) {
        counts[invLookup[i]+1] = counts[invLookup[i]+1]+1; // 0 5 4 6 3
    }
    for(uint32_t j = 1; j < nc+1; ++j) {
        counts[j] = counts[j]+ counts[j-1]; //cumsum 
    }

    //argsort invLookup to get the Lookup
    iota(Lookup, Lookup+nb, 0);
    stable_sort(Lookup, Lookup+nb, [&invLookup](size_t i1, size_t i2) {return invLookup[i1] < invLookup[i2];});

    //save index files
    mkdir(indexpath.c_str(), 0777);
    FILE* f3 = fopen((indexpath+"/dataNorms.bin").c_str(), "wb");
    fwrite(data_norms, sizeof(float), nb, f3);
    fclose (f3);
    FILE* f4 = fopen((indexpath+"/Lookup.bin").c_str(), "wb");
    fwrite(Lookup, sizeof(uint32_t), nb, f4);
    fclose (f4);
    FILE* f5 = fopen((indexpath+"/counts.bin").c_str(), "wb");
    fwrite(counts, sizeof(uint32_t), nc+1, f5);
    fclose (f5);
}   

void BLISS::loadIndex(string indexpath){
    data_norms = new float[nb]{0};
    Lookup= new uint32_t[nb];
    counts = new uint32_t[nc+1]; 

    load(indexpath+"/model.bin");
    FILE* f3 = fopen((indexpath+"/dataNorms.bin").c_str(), "r");
    fread(data_norms, sizeof(float), nb, f3);
    FILE* f4 = fopen((indexpath+"/Lookup.bin").c_str(), "r");
    fread(Lookup, sizeof(uint32_t), nb, f4);
    FILE* f5 = fopen((indexpath+"/counts.bin").c_str(), "r");
    fread(counts, sizeof(uint32_t), nc+1, f5);
    uint32_t emptybins = 0;
    for(uint32_t i = 0; i < nc; ++i) {
        if (counts[i+1]-counts[i]==0) emptybins++;
    }
    cout<<emptybins<<endl;

    // reorder data and index
    dataset_reordered = new float[nb*d];
    data_norms_reordered = new float[nb];
    for(uint32_t i = 0; i < nb; ++i) {
        copy(dataset+Lookup[i]*d, dataset+(Lookup[i]+1)*d , dataset_reordered+i*d);
        data_norms_reordered[i] = data_norms[Lookup[i]];
    }
    delete dataset;
}

uint32_t BLISS::query(float* queryset, int nq, int num_results, int nprobe){
    neighbor_set = new int32_t[nq*num_results]{-1};
    cout<<"num queries: "<<nq<<endl;
    uint32_t seen;
    #pragma omp parallel for
    for (size_t i = 0; i < nq; i++){
        seen += findNearestNeighbor(queryset+(i*d), num_results, nprobe, i);
    }
    return (seen/nq);
}

// start from best cluster -> bruteforce BLISS
uint32_t BLISS::findNearestNeighbor(float* query, int num_results, int nprobe, size_t qnum)
{   
    chrono::time_point<chrono::high_resolution_clock> t0, t1, t2, t3;
    vector<float> topkDist;
    // t0 = chrono::high_resolution_clock::now();
    priority_queue<pair<float, uint32_t> > pq;
    uint32_t simid[nc];
    float simv[nc];

    getscore(query, simv);
    // need argsorted IDs
    iota(simid, simid+nc, 0);
    stable_sort(simid, simid+nc, [&simv](size_t i1, size_t i2) {return simv[i1] > simv[i2];});
    priority_queue<pair<float, uint32_t> > Candidates_pq;
    uint32_t Candidates[nb];
    uint32_t seen=0, seenbin=0;

    float sim;
    float a=0,b=0;
    // t1 = chrono::high_resolution_clock::now();
    for(uint32_t seenbin = 0; seenbin<nprobe; seenbin++){ 
        uint32_t bin = simid[seenbin];
        for (int i =counts[bin]; i< counts[bin+1]; i++){
            Candidates[seen]=i; 
            seen++;
        }
    }
    float score[seen];
    // #pragma omp parallel
    // {
    // int tid = omp_get_thread_num();
    // for (int i =tid*(seen/4); i<(tid+1)*(seen/4); i++){
    //     score[i] = -L2SqrSIMD16ExtAVX(query, dataset_reordered + Candidates[i]*d, data_norms_reordered[Candidates[i]], d);
    //     }
    // }

    for (int i =0; i< seen; i++){
        score[i] = -L2SqrSIMD16ExtAVX(query, dataset_reordered + Candidates[i]*d, data_norms_reordered[Candidates[i]], d);
    }

    // t2 = chrono::high_resolution_clock::now();

    for (int i =0; i< num_results; i++){ 
        Candidates_pq.push({score[i], Candidates[i]});
    }

    float maxk = Candidates_pq.top().first;
    for (int i =num_results; i< seen; i++){ 
        if (score[i]< maxk){
            maxk = Candidates_pq.top().first;
            Candidates_pq.pop();
            Candidates_pq.push({score[i], Candidates[i]});
        }
    }
    for (int i =0; i< num_results; i++){ 
        neighbor_set[qnum*num_results+ i] = Lookup[Candidates_pq.top().second];
        Candidates_pq.pop();
    }
    // t3 = chrono::high_resolution_clock::now();
    // cout<<"hash time: "<<chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count()/10000<<" ";
    // cout<<"dist time: "<<chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count()/10000<<" ";
    // cout<<"rank time: "<<chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count()/10000<<endl;
    return seen;
}


//Improvements
// Remove the empty clusters and their centroids
// give 1 thread per cluster
// merge the few count clusters into one. Only if the clusters are very unbalanced
