// void cgather_batch(long*, long*, long*, long*, long*, double*, int, int, int, int, int);
// void cgather_K(float*, long*, float*, long*, int, int, int, int, int);
 

#include <vector>

class FastIV {
public:
    int k_, m_;
    long N_;
    std::vector<int>* InvIndex;
    FastIV(int k, int m, long N);
    void createIndex(int i, long* list, int L);
    long FC(long* hashIndex, int threshold, int maxsize, long* candidates);
    float distComp(float* query, float* train_data, int d, long* candidates, int cansz, float* ip, int n_threads);
};
