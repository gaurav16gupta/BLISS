# main code for query

from config import eval_config as config
import tensorflow as tf
import time
import numpy as np
import logging
import argparse
import os, sys
import json
import glob
import h5py
from utils import _parse_function
from multiprocessing import Pool
import pdb
sys.path.append('InvertedIndex/')
import scoreAgg

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--node", default=0, type=int)
parser.add_argument("--R", default=4, type=int)
parser.add_argument("--topk", default=25, type=int)
parser.add_argument("--mf", default=2, type=int)
parser.add_argument("--eval_epoch", default=20, type=int)
parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--data", default='deep-1b', type=str)
parser.add_argument("--CppInf", default=1, type=bool)
parser.add_argument("--memmap", default=False, type=bool)
parser.add_argument("--rerank", default=False, type=bool)
dataReorder = False
args = parser.parse_args()

config.logfile = '../logs/'+config.datasetName+'/b_'+str(config.B)+'/node_'+str(args.node)+'/1BR_'+str(args.R)+'_topk_'+str(args.topk)+'_mf_'+str(args.mf)+'_epc_'+str(args.eval_epoch)+'.txt'
config.output_loc = config.logfile[:-3]+'npy'
if args.eval_epoch-5==0:
    config.lookups_loc = config.lookups_loc+'/epoch_0/'
else:
    config.lookups_loc = config.lookups_loc+'/node_'+str(args.node)

if not args.gpu=='all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

part = 0
############################## load model  and defs ##################################
class MyModule(tf.Module):
  def __init__(self, R):
    self.R = R
    self.W1 = [None for r in range(R)]
    self.b1 = [None for r in range(R)]
    self.hidden_layer = [None for r in range(R)]
    self.W2 = [None for r in range(R)]
    self.b2 = [None for r in range(R)]
    self.logits = [None for r in range(R)]
    self.top_buckets = [None for r in range(R)]

  def load(self,paths):
    params = [np.load(path) for path in paths]
    self.W1 = [tf.constant(params[r]['W1']) for r in range(self.R)]
    self.b1 = [tf.constant(params[r]['b1']) for r in range(self.R)]
    self.W2 = [tf.constant(params[r]['W2']) for r in range(self.R)]
    self.b2 = [tf.constant(params[r]['b2']) for r in range(self.R)]

#   @tf.function
#   def __call__(self, next_y_idxs, next_y_vals, next_x_idxs, next_x_vals):
#     x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], next_x_idxs.values], axis=-1),
#         next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(self.R)]
#     x_dense = tf.sparse.to_dense(x[0], validate_indices=False)
#     for r in range(self.R):
#         self.hidden_layer[r] = tf.nn.relu(tf.sparse.sparse_dense_matmul(x_dense[r], self.W1[r])+self.b1[r])
#         self.logits[r] = tf.matmul(self.hidden_layer[r],self.W2[r])+self.b2[r]
#         self.top_buckets[r] = tf.nn.top_k(self.logits[r], k=args.k2, sorted=True)
#     return top_buckets

  @tf.function
  def __call__(self, x):
    for r in range(self.R):
        self.hidden_layer[r] = tf.nn.relu(tf.matmul(x, self.W1[r])+self.b1[r])
        self.logits[r] = tf.matmul(self.hidden_layer[r],self.W2[r])+self.b2[r]
        self.top_buckets[r] = tf.nn.top_k(self.logits[r], k=args.topk, sorted=False) # note the difference here then in indexing
    return self.top_buckets

def process_scores(inp):
    R = inp.shape[0]
    topk = inp.shape[2]
    ##
    # scores = {}
    freqs = {}
    ##
    # print ("h1")
    for r in range(args.R):
        for k in range(topk):
            val = inp[r,0,k] # inp[r,0,k] is values, inp[r,1,k] is the indices
            ##
            for key in inv_lookup[part, r,counts[part, r,int(inp[r,1,k])]:counts[part, r,int(inp[r,1,k])+1]]:
                if key in freqs:
                    # scores[key] += val
                    freqs[key] += 1  
                else:
                    # scores[key] = val
                    freqs[key] = 1
    i = 0
    # print ("h2")
    while True:
        candidates = np.array([key for key in freqs if freqs[key]>=args.mf-i])
        # print (candidates)
        if len(candidates)>=10:
            break
        i += 1
        # print (i)
    ##
    return candidates
    ###

# Preds = np.zeros(args.R*args.topk, dtype=np.int32)
# def process_scorescpp(pt):
#     candid = np.empty(30000, dtype='int32') # does this init takes time?
#     candSize = fastIv.FC(Preds, 30000, pt, candid)
#     return candid[:candSize]

############################## load lookups ################################
# N = config.n_classes
Model = MyModule(args.R)
Model.load([config.model_loc+'/node_0/r_'+str(r)+'_epoch_'+str(args.eval_epoch)+'.npz' for r in range(args.R)])
print ("model loaded")
Parts = 4 # parts per node
block = int(config.n_classes//4)

print (args.CppInf)
if args.CppInf:
    inv_lookup = np.zeros(Parts*args.R*block, dtype=np.int32)
    counts = np.zeros(Parts*args.R*(config.B+1), dtype=np.int32)
    for part in range(Parts):
        for r in range(args.R):
            inv_lookup[(part*args.R + r)*block: (part*args.R + r +1 )*block ] = np.load(config.lookups_loc+'/class_order_'+str(part)+'_'+str(r)+'.npy')# block size 
            counts[(part*args.R + r)*(config.B+1) : (part*args.R + r +1 )*(config.B+1) ] = np.load(config.lookups_loc+'/counts_'+str(part)+'_'+str(r)+'.npy')[:config.B+1] 
    inv_lookup = np.ascontiguousarray(inv_lookup, dtype=np.int32) 
    counts = np.ascontiguousarray(counts, dtype=np.int32)

    fastIv = scoreAgg.PyFastIV(Parts, args.R, block, (config.B+1), args.mf, args.topk, args.node, inv_lookup, counts)
    # fastIv.createIndex() # in future load this directly from a binary file. Saved by C++ code
    print ("Deserialized")

else:
    inv_lookup = np.zeros([Parts, args.R, block], dtype=np.int32)
    counts = np.zeros([Parts, args.R, config.B+1], dtype=np.int32)
    for part in range(Parts):
        for r in range(args.R):
            inv_lookup[part, r] = np.load(config.lookups_loc+'/class_order_'+str(part)+'_'+str(r)+'.npy')# block size 
            counts[part, r] = np.load(config.lookups_loc+'/counts_'+str(part)+'_'+str(r)+'.npy')[:config.B+1] 

# # if dataReorder:
#     reorderDict = np.argsort(inv_lookup[0,:]) #inv map of inv_lookup
#     for r in range(args.R):
#         for b in range(config.B):
#             inv_lookup[r,:] = inv_lookup[r,reorderDict]
#     label_vecs = label_vecs[inv_lookup[0,:]]

################# Data Loader ####################
querypath = config.eval_data_loc + 'queries.npy'
queries = np.load(querypath)
queries = queries[:10000,:]
print("queries loaded ", queries.shape)
queries = tf.data.Dataset.from_tensor_slices(queries)
queries = queries.batch(batch_size = config.batch_size)
iterator = iter(queries)

neighborspath = config.eval_data_loc + 'neighbors100.npy'
neighbors = np.load(neighborspath)

# data load if reranking,  one part is 12GB
dataset = {}
norms = {}
if args.rerank:
    for part in range(Parts):
        # datapath = config.eval_data_loc + 'Deep1Bpart_'+str(args.node*4 + part)+'.h5'
        # dataset[args.node*4 + part]= np.array(h5py.File(datapath, 'r').get('train'))
        datapath = config.eval_data_loc + 'yandex-1b_'+str(args.node*4 + part)+'.dat'

        dataset[args.node*4 + part] = np.array(np.memmap(datapath, dtype='float32', mode='r', shape=(block,config.feat_dim)))
        if config.metric=="L2":
            norms[args.node*4 + part] = np.load(config.eval_data_loc +"normsSq/"+str(args.node*4 + part)+".npy")

    print ("dense vectors leaded")
# to check these
n_check = 10000
count = 0
score_sum = [0.0,0.0,0.0]
output = -1* np.ones([10000,Parts*10])

block_size = (10**9)//32  
#########################################

# p = Pool(config.n_cores)
fw = open(config.logfile, 'a', encoding='utf-8') # log file
bthN = 0
begin_time = time.time()

# p = Pool(Parts)
Inf = 0
RetRank = 0

while True:
    try:
        x_batch = iterator.get_next()
        x_batch = tf.cast(x_batch, tf.float32)
        t1 = time.time()
        top_buckets_ = Model(x_batch) # should give topk bucket IDs, [R,batch_size,topkvals, ]
        top_buckets_ = np.array(top_buckets_)
        top_buckets_ = np.transpose(top_buckets_, (2,0,1,3)) # bring batch_size (index 2) ahead, [batch_size,R,2,topk]
        len_cands = np.zeros(top_buckets_.shape[0])
        t2 = time.time()
        Inf+= (t2-t1)
        for i in range(top_buckets_.shape[0]):
            # ta = time.time()
            
            candidates = []
            if args.CppInf:
                    candid = np.empty(80000, dtype='int32') # does this init takes time?
                    candSize = np.empty(Parts, dtype='int32' )
                    fastIv.FC(np.ascontiguousarray(top_buckets_[i,:,1,:], dtype=np.int32).reshape(-1), 80000, candid, candSize)
                    for part in range(Parts): 
                        candidates.append(candid[(part*80000//Parts):(part*80000//Parts)+ candSize[part]])
                # Below one doesn't work
                # Preds = np.ascontiguousarray(top_buckets_[i,:,1,:], dtype=np.int32).reshape(-1)
                # candidates = p.map(process_scorescpp, [part for part in range(Parts)]) # parallel
            else:
                for part in range(Parts): 
                    candidates.append(process_scores(top_buckets_[i]))
            # tb = time.time()
            # print('Ret: ',(tb-ta))
            # reranking
            if args.rerank:
                for part in range(Parts): 
                    start_lbl = (args.node*4 + part)*block_size 
                    if config.metric == "ip":
                        dists = np.dot(dataset[args.node*4 + part][candidates[part]],x_batch[i]) # or L2 dist
                    if config.metric == "L2":
                        dists = 2*np.dot(dataset[args.node*4 + part][candidates[part]],x_batch[i]) -norms[args.node*4 + part][candidates[part]]
                    if len(dists)<=10:
                        output[bthN*config.batch_size + i, part*10: part*10 + len(dists)] = candidates[part] + start_lbl
                    else:
                        top_cands = np.argpartition(dists, -10)[-10:]
                        output[bthN*config.batch_size + i, part*10: (part+1)*10] = candidates[part][top_cands] + start_lbl
                    # score_sum[0] += len(np.intersect1d(output[bthN*config.batch_size + i, part*10: (part+1)*10], neighbors[i,:10]))/10

                    # len_cands[i] += len(candidates[part])
            else:
                for part in range(Parts): 
                    start_lbl = (args.node*4 + part)*block_size 
                    candidates[part] = candidates[part] + start_lbl

                score_sum[0] += len(np.intersect1d(candidates[part], neighbors[i,:10]))/10
                len_cands[i] += len(candidates[part])
            # pdb.set_trace()
            # tc = time.time()
            # print('Rank: ',(tc-tb))
            score_sum[1] += np.sum(len_cands)
            # pdb.set_trace()
            # shortlist = p.map(process_scores, top_buckets_) # parallel over points in the batch, 2 D array
        t3 = time.time()
        RetRank+= t3-t2
        bthN+=1
        print (bthN)
    except:
        print (bthN)
        print ("node: ", args.node, " topk: ", args.topk, " mf: ", args.mf)
        print('overall Recall for',count,'points:',score_sum[0]/(bthN*config.batch_size + i))
        print('Avg can. size for',count,'points:',score_sum[1]/(bthN*config.batch_size + i))
        print('Inf per point: ',Inf/(bthN*config.batch_size))
        print('Ret+rank per point: ',RetRank/(bthN*config.batch_size))
        print('per point to report: ',(Inf/32 + RetRank/4)/(bthN*config.batch_size))

        print ("node: ", args.node, " topk: ", args.topk, " mf: ", args.mf, file=fw)
        print('overall Recall for',count,'points:',score_sum[0]/(bthN*config.batch_size + i), file=fw)
        print('Avg can. size for',count,'points:',score_sum[1]/(bthN*config.batch_size + i), file=fw)
        print('Inf per point: ',Inf/(bthN*config.batch_size), file=fw)
        print('Ret+rank  per point: ',RetRank/(bthN*config.batch_size), file=fw)
        print('per point to report: ',(Inf/32 + RetRank/4)/(bthN*config.batch_size), file=fw)
        np.save(config.output_loc,output)
        break

# p.close()

