from config import config
import tensorflow as tf
import time
import numpy as np
import argparse
import os
import pdb
import sys
from dataPrepare import *
from net import MyModule

parser = argparse.ArgumentParser()
parser.add_argument("--index", default='glove_epc20_K2_B4096_R4', type=str)
args = parser.parse_args()
datasetName = args.index.split('_')[0]  
n_epochs = int(args.index.split('_')[1].split('epc')[1]) 
K = int(args.index.split('_')[2].split('K')[1])  
B = int(args.index.split('_')[3].split('B')[1])
R = int(args.index.split('_')[4].split('R')[1])

def Index(B,R,datasetName, load_epoch, K):
    bucketSort = True
    # if not gpu=='all':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #########################Tochange
    model_save_loc = "../indices/{}/".format(datasetName)
    lookups_loc  = "../indices/{}/".format(datasetName)
    N = config.DATASET[datasetName]['N'] 
    train_data_loc = "../../data/{}/".format(datasetName)
    batch_size = 5000

    ##########################
    # N = n_classes

    Model = MyModule(R)
    Model.load([model_save_loc+'/r_'+str(r)+'.npz' for r in range(R)]) # node 0 for all
    print ("model loaded")
    # print (lookups_loc+'epoch_'+str(load_epoch))

    datapath = train_data_loc 
    dataset = tf.data.Dataset.from_tensor_slices(getFulldata(datasetName, datapath).astype(np.float32))
    dataset = dataset.batch(batch_size = batch_size)
    iterator = iter(dataset)
    print("data loaded")

    top_preds = np.zeros([R, N, K], dtype=np.int32)

    # p = Pool(n_cores)
    t1 = time.time()
    # pdb.set_trace()
    start_idx = 0
    while True: # this loops for batches
        try:
            # print (start_idx)
            top_preds[:, start_idx:min(start_idx+batch_size, N)]  = Model(iterator.get_next(), K) # should give top K bucket IDs
            start_idx += batch_size
            sys.stdout.write("Inference progress: %d%%   \r" % (start_idx*100/N) )
            sys.stdout.flush()
        except:
            print (start_idx)
            # pdb.set_trace()
            assert (start_idx >=N), "batch iterator issue!"
            break

    t2 = time.time()
    print("Inference time: ", t2-t1)
    #####################################
    try:
        #make it parallel
        for r in range(R):
            counts = np.zeros(B+1, dtype=np.int32)
            bucket_order = np.zeros(N, dtype=np.int32)
            for i in range(N):
                bucket = top_preds[r, i, np.argmin(counts[top_preds[r,i]+1])] 
                bucket_order[i] = bucket
                counts[bucket+1] += 1  
                        
            counts = np.cumsum(counts)
            class_order = np.zeros(N,dtype=np.int32)
            class_order = np.argsort(bucket_order)
            # sorting buckets
            if bucketSort:
                for b in range(B):
                    class_order[counts[b]:counts[b+1]] = np.sort(class_order[counts[b]:counts[b+1]])
            ###
            folder_path = lookups_loc+'epoch_'+str(load_epoch)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path+'/class_order_'+str(r)+'.npy', class_order)
            np.save(folder_path+'/counts_'+str(r)+'.npy', counts)
            np.save(folder_path+'/bucket_order_'+str(r)+'.npy', bucket_order)
            print (r)
    except:
        print ("check indexing issue", r)
    t3 = time.time()
    print("indexed and saved in time: ", t3-t2)

Index(B, R, datasetName, n_epochs, K)
