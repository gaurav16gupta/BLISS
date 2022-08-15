
import tensorflow as tf
import time
import numpy as np
import os, sys
import pdb
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from multiprocessing import Pool
from sklearn.utils import murmurhash3_32 as mmh3


def savememmap(path, ar):
    if path[-4:]!='.dat':
        path = path +'.dat'
    shape = ar.shape
    dtype = ar.dtype
    fp = np.memmap( path, dtype=dtype, mode='w+', shape=(shape))
    fp[:]= ar[:]
    fp.flush()

def getTrueNNS(x_train, metric, K):
    begin_time = time.time()
    batch_size = 1000
    output = np.zeros([x_train.shape[0], K], dtype=np.int32) # for upto 2B

    if metric=='IP':
        W = x_train.T
        for i in range(x_train.shape[0]//batch_size):
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            x_batch = x_train[start_idx:end_idx]
            sim = x_batch@W
            top_idxs = np.argpartition(sim, -K)[:,-K:]
            output[start_idx:end_idx] = top_idxs

    elif metric=='L2':
        W = x_train.T
        W_norm = np.square(np.linalg.norm(W,axis=0))
        for i in range(x_train.shape[0]//batch_size):
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            x_batch = x_train[start_idx:end_idx]
            sim = 2*x_batch@W - W_norm
            top_idxs = np.argpartition(sim, -K)[:,-K:]
            output[start_idx:end_idx] = top_idxs

    elif metric=='cosine':
        x_train = x_train/(np.linalg.norm(x_train,axis=1)[:,None])
        W = x_train.T
        for i in range(x_train.shape[0]//batch_size):
            # t1 = time.time()
            start_idx = i*batch_size
            end_idx = start_idx+batch_size
            x_batch = x_train[start_idx:end_idx]
            sim = x_batch@W # tf this
            top_idxs = np.argpartition(sim, -K)[:,-K:] # use tf.nn.topk It uses mul cores
            output[start_idx:end_idx] = top_idxs
            # print (i, ': ', time.time()-t1)
    print(time.time()-begin_time)
    return output


def create_universal_lookups(r, B, n_classes, lookups_loc):
    c_o = lookups_loc+'class_order_'+str(r)+'.npy'
    ct = lookups_loc+'counts_'+str(r)+'.npy'
    b_o = lookups_loc+'bucket_order_'+str(r)+'.npy'
    if os.path.exists(c_o) and os.path.exists(ct) and os.path.exists(b_o):
        print ('init lookups exists')
    else:
        counts = np.zeros(B+1, dtype=int)
        bucket_order = np.zeros(n_classes, dtype=int)
        for i in range(n_classes):
            bucket = mmh3(i,seed=r)%B
            bucket_order[i] = bucket
            counts[bucket+1] += 1
        counts = np.cumsum(counts)
        rolling_counts = np.zeros(B, dtype=int)
        class_order = np.zeros(n_classes,dtype=int)
        for i in range(n_classes):
            temp = bucket_order[i]
            class_order[counts[temp]+rolling_counts[temp]] = i
            rolling_counts[temp] += 1
        
        np.save(c_o, class_order)
        np.save(ct,counts)
        np.save(b_o, bucket_order)

# to do: fix this
def process_scores(inp, ):
    R = inp.shape[0]
    topk = inp.shape[2]
    # scores = {}
    freqs = {}
    for r in range(R):
        for k in range(topk):
            val = inp[r,0,k] # inp[r,0,k] is values, inp[r,1,k] is the indices
            for key in inv_lookup[r,counts[r,int(inp[r,1,k])]:counts[r,int(inp[r,1,k])+1]]:
                if key in freqs:
                    # scores[key] += val
                    freqs[key] += 1  
                else:
                    # scores[key] = val
                    freqs[key] = 1
    i = 0
    while True:
        candidates = np.array([key for key in freqs if freqs[key]>=args.mf-i])
        if len(candidates)>=10:
            break
        i += 1
    return candidates
    ###