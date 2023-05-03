#include "utils.h"

using namespace std;

double RecallAtK(int* answer, int* guess, size_t k, size_t nq){
    uint32_t count = 0;
    for (int i=0;i<nq;i++){
        sort(answer+ k*i, answer + (i+1)*k);
        sort(guess+ k*i, guess+ (i+1)*k);
        std::vector<int> tmp;
        std::set_intersection(answer+ k*i, answer + (i+1)*k,  // Input iterators for first range 
                            guess+ k*i, guess+ (i+1)*k, // Input iterators for second range 
                            std::back_inserter(tmp));
        count += double(tmp.size());
    }
    return (count/double(nq*k));
}

float L2sim(float* a, float* b, float norm_bsq, size_t d){
    float sim=0;
    for(uint32_t k = 0; k < d; ++k) {    
        sim += a[k]*b[k]; // one unit FLOP- mul
        // sim += pow(a[k]-b[k],2); // two units FLOPS- mul and sub
    } 
    sim= sim- norm_bsq;
    return sim;
}

// Not giving speedup!! Check the issue
float L2SqrSIMD16ExtAVX(float *pVect1, float *pVect2, float norm_bsq, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty16 = qty / 16;
    const float *pEnd1 = pVect1 + 16 * qty16;
    __m256 sum256 = _mm256_set1_ps(0);
    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        const __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        const __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }
    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] - norm_bsq;
    return sum;
}

float IP(float* a, float* b, size_t d){
    float ip=0;
    for(uint32_t k = 0; k < d; ++k) {    
        ip += a[k]*b[k]; // one unit FLOP- mul
    } 
    return ip;
}
        
float IPSIMD16ExtAVX(float *pVect1, float *pVect2, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty / 16;
    const float *pEnd1 = pVect1 + 16 * qty16;
    __m256 sum256 = _mm256_set1_ps(0);
    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        const __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        const __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }
    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    return sum;
}

uint16_t getclusterPart(uint16_t* maxMC, vector<uint16_t> &props, int treelen){
    // maxMC: property location, property, frequency
    for (uint16_t i=0;i<treelen; i++){
        if (maxMC[i*3+1] == props[maxMC[i*3+0]]){
            return i;
        }
    }
    return treelen;    
}

//checks if the property x is seen before in maxMC
bool not_in(uint16_t x, uint16_t* a, int h){
    // property location, property, frequency
    if (h == 0){
        return 1;
    } 
    else{
        for(uint16_t i=0;i< h;i++){ 
            if (a[i*3+1]==x){return 0;}
        };
        return 1;
    }
}

void randomShuffle(int* v , int l, int u){
     // Range of numbers [l, u]
    iota(v, v+u-l, l); 
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v, v+u-l, g);
}


vector<uint32_t> argTopK(float* query, float* vectors, uint32_t d, uint32_t N, vector<uint32_t> idx, uint32_t idxSize, uint32_t k, vector<float> topkDist){
    float dist; 
    vector<uint32_t> topk;
    priority_queue<pair<float, uint32_t> > pq;
    if (idxSize ==N){
        for (uint32_t i = 0; i < N; i++){
            //L2
            dist =0;
            for (size_t j = 0; j < d; j++){
                dist += pow(vectors[i*d+j] - query[j], 2);
            }
            dist = sqrt(dist);
            //topk
            if (i<k) pq.push({dist, i});
            else{
                if (dist< pq.top().first){
                    pq.pop();
                    pq.push({dist, i});
                }
            }
        }
    }
    else{
        for (uint32_t i = 0; i < idxSize; i++){
            //L2
            try{
                dist =0;
                for (size_t j = 0; j < d; j++){
                    dist += pow(vectors[idx[i]*d+j] - query[j], 2);//*
                }

                dist = sqrt(dist);
                //topk
                if (i<k) pq.push({dist, idx[i]});
                else{
                    if (dist< pq.top().first){
                        pq.pop();
                        pq.push({dist, idx[i]});
                    }
                }
            }
            catch(int mynum){
                cout << "Error number: "; 
            }
        }
    }
    for (uint32_t i = 0; i < k; i++){
        topk.push_back(pq.top().second);
        topkDist.push_back(pq.top().first);
        pq.pop();
    }
    return topk;
}




/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}
