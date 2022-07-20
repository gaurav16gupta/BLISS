
#include <vector>

class FastIV {
public:
    int Parts_, R_, B_, threshold, topk_, node_;
    int block_;
    int* inv_lookup_;
    int* counts_;
    int tempe;
    uint8_t* counter; 
    // std::vector<int>* InvIndex;

    FastIV(int Parts, int R, int block, int B, int mf, int topk, int node, int* inv_lookup, int* counts);
    // void createIndex(int i, long* list, int L);
    void FC(int* top_buckets_, int maxsize, int* candidates, int* candSize);
};
