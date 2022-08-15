// void cgather_batch(long*, long*, long*, long*, long*, double*, int, int, int, int, int);
// void cgather_K(float*, long*, float*, long*, int, int, int, int, int);
 

#include <vector>

class FastIV {
public:
    int R_, B_, threshold, topk_;
    int block_;
    int* inv_lookup_;
    int* counts_;
    int tempe;
    uint8_t* counter; 
    // std::vector<int>* InvIndex;

    FastIV(int R, int block, int B, int mf, int topk, int* inv_lookup, int* counts);
    // void createIndex(int i, long* list, int L);
    void FC(int* top_buckets_, int maxsize, int* candidates, int* candSize);
};
