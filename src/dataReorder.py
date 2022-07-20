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
parser.add_argument("--CppInf", default=1, type=bool)
# parser.add_argument("--R_per_gpu", default=16, type=int)
parser.add_argument("--rerank", default=False, type=bool)

args = parser.parse_args()
B = 65536
Nodes = 8
Parts = 4
dataset ={}
N = (10**9)//(Nodes*Parts)

r = 0
datasetName ="yandex"
d = 200
dtype = 'float32'

lookups_loc = '../lookups/'+datasetName+'/b_'+str(B)+'/' # '/epoch_'+str(eval_epoch-5)+'/node_'+str(node)
eval_data_loc = '../../data/'+datasetName+'/'
t1 = time.time()
for node in range(Nodes):
    for part in range(Parts):
        datapath = eval_data_loc + 'yandex-1b_'+str(node*4 + part)+'.dat'
        dataset[node*4 + part] = np.memmap(datapath, dtype=dtype, mode='r', shape=(N,d))
        print ("dense vectors loaded")

        inv_lookup = []
        for r in range(args.R):
            inv_lookup.append(np.load(lookups_loc+'/node_'+str(node)+'/class_order_'+str(part)+'_'+str(r)+'.npy'))# block size 

        # if dataReorder:
        reorderDict = np.argsort(inv_lookup[0]) #inv map of inv_lookup
        for r in range(args.R):
             # input old idex ..get new index
            np.save(lookups_loc+'/reordered/node_'+str(node)+'/class_order_'+str(part)+'_'+str(r)+'.npy', reorderDict[inv_lookup[r]])

        datapath = eval_data_loc + 'reordered/yandex-1b_'+str(node*4 + part)+'.dat'
        fp = np.memmap(datapath, dtype=dtype, mode='w+', shape=(N,d))
        fp[:]= dataset[node*4 + part] = dataset[node*4 + part][inv_lookup[0],:][:]
        fp.flush()

        #key
        np.save(lookups_loc+'/reordered/node_'+str(node)+'/reorderKey_'+str(part)+'.npy', inv_lookup[0] )

print (time.time()-t1)