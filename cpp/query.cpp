#include <iostream>
#include <fstream>
#include "BLISS.h"

int main(int argc, char** argv)
{
    if (argc != 4){
        std::cout << argv[0] << " dataname num_clusters nprobe"<< std::endl;
        exit(-1);
    }
    
    string datapath = "../../../data/" + string(argv[1]) + "/base.fvecs"; 
    string querypath = "../../../data/" + string(argv[1]) + "/query.fvecs"; 
    string GTpath = "../../../data/" + string(argv[1]) +"/"+ string(argv[1]) + "_groundtruth.ivecs"; 
    string indexpath = "indices/"+string(argv[1])+ "Mode1/";
    
    string metric;
    if (string(argv[1])=="sift"){
        metric="L2";}
    if (string(argv[1])=="glove"){
        metric="Angular";}
    else{
        metric="L2";
    }

    size_t d, nb,nc, nq, num_results, nprobe; 
    float* data = fvecs_read(datapath.c_str(), &d, &nb);
    nc = atoi(argv[2]); // num clusters
    BLISS mysearch(data, d, nb, nc);
    mysearch.loadIndex(indexpath);
    cout << "Loaded" << endl;

    float* queryset = fvecs_read(querypath.c_str(), &d, &nq);
    int* queryGTlabel = ivecs_read(GTpath.c_str(), &num_results, &nq);
    cout << "Query files read..." << endl;
    nq = 10000;
    nprobe = atoi(argv[3]);
    
    chrono::time_point<chrono::high_resolution_clock> t1, t2;
    t1 = chrono::high_resolution_clock::now();
    uint32_t numDist = mysearch.query(queryset, nq, num_results, nprobe);
    t2 = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    
    int32_t* output = mysearch.neighbor_set;
    int output_[num_results*nq];
    copy(output, output+num_results*nq , output_);
    cout<<"numClusters, buffersize, QPS, Recall100@100 :"<<endl;
    //QPS and recall
    double QPS;
    double recall = RecallAtK(queryGTlabel, output_, num_results, nq);
    printf("%d,%d,%d,%d,%f\n",nc, nprobe, numDist, (int)(nq/diff.count()), recall);
}
