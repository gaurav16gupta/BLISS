// after training 
#include <iostream>
#include <fstream>
#include "BLISS.h"

int main(int argc, char** argv)
{
    if (argc != 3){
        std::cout << argv[0] << " dataname num_bins"<< std::endl;
        exit(-1);
    }
    string datapath = "../../../data/" + string(argv[1]) + "/base.fvecs"; 
    string indexpath = "indices/"+string(argv[1])+ "Mode1/";

    string metric;
    if (string(argv[1])=="sift"){
        metric="L2";}
    if (string(argv[1])=="glove"){
        metric="Angular";}
    else{
        metric="L2";
    }
    // get BLISS training, or any other NN model 
    size_t d, nb,nc; 
    float* data = fvecs_read(datapath.c_str(), &d, &nb);
    nc = atoi(argv[2]); // num clusters
    BLISS mysearch(data, d, nb, nc);
    cout<<indexpath<<endl;
    mysearch.make_index(indexpath, "BLISS");

}
