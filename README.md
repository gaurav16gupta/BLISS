# BLISS: A Billion scale Index using Iterative Re-partitioning


ThirdAI Corp., Texas, USA
**Authors**: [Gaurav Gupta](https://gaurav16gupta.github.io/), [Tharun Medini](https://tharun24.github.io/), [Anshumali Shrivastava](https://www.cs.rice.edu/~as143/), [Alexander J. Smola](https://alex.smola.org/)

## Abstract

Representation learning has transformed the problem of information retrieval into one of finding the approximate set of nearest neighbors in a high dimensional vector space. With limited hardware resources and time-critical queries, the retrieval engines face an inherent tension between latency, accuracy, scalability, compactness, and the ability to load balance in distributed settings. To improve the trade-off, we propose a new algorithm, called BaLanced Index for Scalable Search (BLISS), a highly tunable indexing algorithm with enviably small index sizes, making it easy to scale
to billions of vectors. It iteratively refines partitions of items by learning the relevant buckets directly from the query-item relevance data. To ensure that the buckets are balanced, BLISS uses the power-of-𝐾 choices strategy. We show that BLISS provides superior load balancing with high probability (and under very benign assumptions). Due to its design, BLISS can be employed for both near-neighbor retrieval (ANN problem) and extreme classification
(XML problem). For the case of ANN, we train and index 4 datasets with billion vectors each. We compare the recall, inference time, indexing time, and index size for BLISS with the two most popular and well-optimized libraries- Hierarchical Navigable Small World (HNSW) graph and Facebook’s FAISS. BLISS requires 100× lesser RAM than HNSW, making it fit in memory on commodity machines while taking a similar inference time as HNSW for the same recall. Against FAISS-IVF, BLISS achieves similar performance with 3-4× less memory requirement. BLISS is both data and model parallel, making it ideal for distributed implementation for training and inference. For the case of XML, BLISS surpasses the best baselines’ precision while being 5× faster for inference on popular multi-label datasets with half a million classes.

## Prerequisites
```
$ pip install -r requirements.txt
```

## Run
* Train the iterative model
```
$ python3 construct.py --index='glove_epc20_K2_B4096_R4'
```
* Index on the trained model
```
$ python3 index.py --index='glove_epc20_K2_B4096_R4'
```
* Query the index
```
$ python3 query.py --index='glove_epc20_K2_B4096_R4' --topm=50
```


## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. 

## Acknowledgment
The code is build upon [Tharun24/MACH](https://github.com/Tharun24/MACH)

## Citation
If you found the provided code useful, please cite our work with the following bibtex.

```bibtex
@inproceedings{10.1145/3534678.3539414,
author = {Gupta, Gaurav and Medini, Tharun and Shrivastava, Anshumali and Smola, Alexander J.},
title = {BLISS: A Billion Scale Index Using Iterative Re-Partitioning},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539414},
doi = {10.1145/3534678.3539414},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {486–495},
numpages = {10},
keywords = {classification, search, load-balance, billion-scale, learning-to-index},
location = {Washington DC, USA},
series = {KDD '22}
}
```



### Requirements: tensorflow 2.4



Get datasets from

https://big-ann-benchmarks.com/index.html#call

https://github.com/erikbern/ann-benchmarks

http://manikvarma.org/downloads/XC/XMLRepository.html

