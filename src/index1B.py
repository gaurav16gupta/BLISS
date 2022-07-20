# from config import eval_config as config
from config import train_config as config
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import time
import numpy as np
import logging
import argparse
import os
import json
import glob
import h5py
from utils import _parse_function
from multiprocessing import Pool
import argparse
import pdb
import sys

# it will take the learned model and index the part of 1B vectors in a lookup table
parser = argparse.ArgumentParser()
parser.add_argument("--node", default=0, type=int)
parser.add_argument("--R", default=4, type=int)
parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--R_per_gpu", default=4, type=int)
parser.add_argument("--load_epoch", default=20, type=int)
parser.add_argument("--k2", default=2, type=int, help='take top-k2 buckets of accumulated label vectors and assign it to the least occupied')

bucketSort = True
dataReorder = False
args = parser.parse_args()

# config.logfile = '../logs/'+config.datasetName+'/b_'+str(config.B)+'/node_'+str(args.node)+'/IndexingR_'+str(args.R)+'_epc_'+str(args.load_epoch)+'.txt'
# config.output_loc = config.logfile[:-3]+'npy' 

# if args.load_epoch-5==0:
#     config.lookups_loc = '../lookups/'+config.datasetName+'/b_'+str(config.B)+'/epoch_0/'
# else:
#     config.lookups_loc = '../lookups/'+config.datasetName+'/b_'+str(config.B)+'/epoch_'+str(args.load_epoch-5)+'/node_'+str(args.node)+'/'

if not args.gpu=='all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
        self.top_buckets[r] = tf.nn.top_k(self.logits[r], k=args.k2, sorted=False)[1]
    return self.top_buckets

N = config.n_classes

Model = MyModule(args.R)
Model.load([config.model_save_loc+'node_0/r_'+str(r)+'_epoch_'+str(args.load_epoch)+'.npz' for r in range(args.R)]) # node 0 for all
print ("model loaded")
print (config.lookups_loc+'ep'+str(args.load_epoch)+'/node_'+str(args.node))
Parts = 4 # parts per node
block = int(config.n_classes//4)
# write numpy iterator

for part in range(Parts):
    # datapath = config.train_data_loc + 'Deep1Bpart_'+str(args.node*4 + part)+'.dat'
    # dataset = tf.data.Dataset.from_tensor_slices(np.array(np.memmap(datapath, dtype='float32', mode='c', shape=(10**9//32,96))))
    # dataset = tf.data.Dataset.from_tensor_slices(np.array(np.memmap(datapath, dtype='uint8', mode='c', shape=(10**9//32,config.feat_dim))).astype(np.float32))
    datapath = config.train_data_loc + 'Deep1Bpart_'+str(args.node*4 + part)+'.dat'
    dataset = tf.data.Dataset.from_tensor_slices(np.array(np.memmap(datapath, dtype='float32', mode='c', shape=(10**9//32,config.feat_dim))).astype(np.float32))
    dataset = dataset.batch(batch_size = config.batch_size)
    iterator = iter(dataset)
    print("data loaded")
    # print (iterator.get_next())
    # iterator = dataset.make_initializable_iterator()
    # next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
    # pdb.set_trace()
    # next_y_idxs, next_y_vals, next_x_idxs, next_x_vals  = dataset.take(config.batch_size)
    # x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup[r], next_x_idxs.values)], axis=-1),
    # next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(args.R)]

    # x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], next_x_idxs.values], axis=-1), next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(args.R)]
    # x_dense = tf.sparse_tensor_to_dense(x[0], validate_indices=False)

    ############################## Create Graph ################################

    ##### Run Graph Optimizer on first batch (might take ~50s) ####
    # sess.run(iterator.initializer)
    # top_preds, y_idxs = sess.run([top_buckets, next_y_idxs])
    top_preds = np.zeros([args.R, block, args.k2], dtype=np.int32)

    # debug
    # print (iterator.get_next())
    ###### Re-initialize the data loader ####
    # sess.run(iterator.initializer)

    # p = Pool(config.n_cores)
    t1 = time.time()
    # pdb.set_trace()
    start_idx = 0
    while True: # this loops for batches
    # for data in dataset:
        # a,b,c,d = next(iterator)
        
        # next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
        # next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = dataset.take(1)  
        # top_preds[:, start_idx:start_idx+config.batch_size]  = Model(next_y_idxs, next_y_vals, next_x_idxs, next_x_vals) # should give top k2 bucket IDs
        try:
            top_preds[:, start_idx:min(start_idx+config.batch_size, block)]  = Model(iterator.get_next()) # should give top k2 bucket IDs
            start_idx += config.batch_size
            sys.stdout.write("Inference progress: %d%%   \r" % (start_idx*100/block) )
            sys.stdout.flush()
        except:
            print (start_idx)
            # pdb.set_trace()
            assert (start_idx >=block), "batch iterator issue!"
            break
        # x = [tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], next_x_idxs.values], axis=-1),
        # next_x_vals.values, [config.batch_size, config.feat_hash_dim]) for r in range(self.R)]
        # x_dense = tf.sparse_tensor_to_dense(x[0], validate_indices=False)
        # top_preds[:, start_idx:start_idx+config.batch_size]  = Model(x_dense) # should give top k2 bucket IDs

    t2 = time.time()
    print("Inference time: ", t2-t1)
    #####################################
    try:
        for r in range(args.R):
            counts = np.zeros(config.B+1, dtype=np.int32)
            bucket_order = np.zeros(block, dtype=np.int32)
            for i in range(block):
                bucket = top_preds[r, i, np.argmin(counts[top_preds[r,i]+1])] 
                bucket_order[i] = bucket
                counts[bucket+1] += 1  
                      
            ###
            # nothing = sess.run(tf.assign(lookup,bucket_order)) # why?
            ###
            counts = np.cumsum(counts)
            class_order = np.zeros(block,dtype=np.int32)
            class_order = np.argsort(bucket_order)
            # rolling_counts = np.zeros(config.B, dtype=np.int32)
            # for i in range(block):
            #     temp = bucket_order[i]
            #     class_order[counts[temp]+rolling_counts[temp]] = i
            #     rolling_counts[temp] += 1

            # sorting buckets
            if bucketSort:
                for b in range(config.B):
                    class_order[counts[b]:counts[b+1]] = np.sort(class_order[counts[b]:counts[b+1]])
            ###
            folder_path = config.lookups_loc+'ep'+str(args.load_epoch)+'/node_'+str(args.node)
            # if not os.path.isdir(folder_path):
            #     os.system('mkdir '+folder_path)
            np.save(folder_path+'/class_order_'+str(part)+'_'+str(r)+'.npy', class_order)
            np.save(folder_path+'/counts_'+str(part)+'_'+str(r)+'.npy', counts)
            np.save(folder_path+'/bucket_order_'+str(part)+'_'+str(r)+'.npy', bucket_order)
    except:
        print ("check indexing issue", part, r)
    t3 = time.time()
    print("indexed and saved in time: ", t3-t2)

# p.close()

