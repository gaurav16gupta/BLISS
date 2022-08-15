from config import config
import tensorflow as tf
import time
import numpy as np
import argparse
import os, sys
from utils import *
from multiprocessing import Pool
import pdb
sys.path.append('InvertedIndex/')
import scoreAgg
from net import MyModule
from dataPrepare import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--topm", default=10, type=int)
parser.add_argument("--mf", default=2, type=int)
parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--index", default='deep-1b_epc20_K2_B65536_R4', type=str)
# parser.add_argument("--CppInf", default=1, type=bool)
parser.add_argument("--memmap", default=False, type=bool)
parser.add_argument("--rerank", default=True, type=bool)
args = parser.parse_args()

datasetName = args.index.split('_')[0]  
eval_epoch = int(args.index.split('_')[1].split('epc')[1]) 
K = int(args.index.split('_')[2].split('K')[1])  
B = int(args.index.split('_')[3].split('B')[1])
R = int(args.index.split('_')[4].split('R')[1])
feat_dim =  config.DATASET[datasetName]['d']
N = config.DATASET[datasetName]['N'] 
metric = config.DATASET[datasetName]['metric'] 
dtype = config.DATASET[datasetName]['dt'] 
lookups_loc  = "../indices/{}/".format(datasetName) + '/epoch_'+ str(eval_epoch)
model_loc = "../indices/{}/".format(datasetName)
data_loc = "../../data/{}/".format(datasetName)
buffer = 1024*(int(2*R*N*args.topm/(B*args.mf))//1024)

batch_size = 32
logfile = '../logs/'+datasetName+'/'+args.index+'query.txt'
output_loc = logfile[:-3]+'npy'

if not args.gpu=='all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

############################## load lookups ################################
Model = MyModule(R)
Model.load([model_loc+'/r_'+str(r)+'_epoch_'+str(eval_epoch)+'.npz' for r in range(R)])
print ("model loaded")

inv_lookup = np.zeros(R*N, dtype=np.int32)
counts = np.zeros(R*(B+1), dtype=np.int32)
for r in range(R):
    inv_lookup[r*N: (r +1)*N ] = np.load(lookups_loc+'/class_order_'+str(r)+'.npy')# block size 
    counts[r*(B+1) : (r +1 )*(B+1) ] = np.load(lookups_loc+'/counts_'+str(r)+'.npy')[:B+1] 
inv_lookup = np.ascontiguousarray(inv_lookup, dtype=np.int32) 
counts = np.ascontiguousarray(counts, dtype=np.int32)

fastIv = scoreAgg.PyFastIV(R, N, (B+1), args.mf, args.topm, inv_lookup, counts)
# fastIv.createIndex() # in future load this directly from a binary file. Saved by C++ code
print ("Deserialized")

################# Data Loader ####################
[queries, neighbors100] = getQueries(datasetName)
queries = queries[:1000,:]
print("queries loaded ", queries.shape)
queries = tf.data.Dataset.from_tensor_slices(queries)
queries = queries.batch(batch_size = batch_size)
iterator = iter(queries)

if args.rerank:
    datapath = data_loc +'fulldata.dat'
    dataset = getFulldata(datasetName, datapath)
    if metric=="L2":
        norms= np.load(data_loc +"norms.npy")
    if metric =="cosine":
        norms= np.load(data_loc +"norms.npy")
        dataset = dataset/(norms[:,None])
    print ("dense vectors loaded")
# to check these
count = 0
score_sum = [0.0,0.0,0.0]
output = -1* np.ones([10000,10])
#########################################

# p = Pool(config.n_cores)
fw = open(logfile, 'a', encoding='utf-8') # log file
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
        top_buckets_ = Model(x_batch, args.topm) # should give topm bucket IDs, [R,batch_size,topmvals, ]
        top_buckets_ = np.array(top_buckets_)
        # top_buckets_ = np.transpose(top_buckets_, (2,0,1,3)) # bring batch_size (index 2) ahead, [batch_size,R,2,topm]
        top_buckets_ = np.transpose(top_buckets_, (1,0,2)) # bring batch_size (index 1) ahead, [batch_size,R,topm]

        len_cands = np.zeros(top_buckets_.shape[0])
        t2 = time.time()
        Inf+= (t2-t1)
        for i in range(top_buckets_.shape[0]):
            candid = np.empty(buffer, dtype='int32') # does this init takes time?
            candSize = np.empty(1, dtype='int32' )
            fastIv.FC(np.ascontiguousarray(top_buckets_[i,:,:], dtype=np.int32).reshape(-1), buffer, candid, candSize)
            candidates = (candid[0: candSize[0]])
            
            # candidates = (process_scores(top_buckets_[i]))
            score_sum[1] += len(candidates)
            if args.rerank:
                if metric == "IP":
                    dists = np.dot(dataset[candidates],x_batch[i]) # or L2 dist
                if metric == "L2":
                    dists = 2*np.dot(dataset[candidates],x_batch[i]) -norms[candidates]
                if metric =="cosine":
                    dists = np.dot(dataset[candidates],x_batch[i]) # or L2 dist
                if len(dists)<=10:
                    output[bthN*batch_size + i, :len(dists)] = candidates 
                if len(dists)>10:
                    top_cands = np.argpartition(dists, -10)[-10:]
                    output[bthN*batch_size + i, :10] = candidates[top_cands] 
                    candidates = candidates[top_cands] 
                
            score_sum[0] += len(np.intersect1d(candidates, neighbors100[bthN*batch_size + i,:10]))/10

        t3 = time.time()
        RetRank+= t3-t2
        bthN+=1
        print (bthN)
    except:
        # print (bthN)
        print ( " topm: ", args.topm, " mf: ", args.mf)
        print('overall Recall for',count,'points:',score_sum[0]/((bthN-1)*batch_size + i))
        print('Avg can. size for',count,'points:',score_sum[1]/((bthN-1)*batch_size + i))
        pdb.set_trace()
        print('Inf per point: ',Inf/((bthN-1)*batch_size))
        print('Ret+rank per point: ',RetRank/((bthN-1)*batch_size))
        print('per point to report: ',(Inf/32 + RetRank/4)/((bthN-1)*batch_size))

        print (" topm: ", args.topm, " mf: ", args.mf, file=fw)
        print('overall Recall for',count,'points:',score_sum[0]/((bthN-1)*batch_size + i), file=fw)
        print('Avg can. size for',count,'points:',score_sum[1]/((bthN-1)*batch_size + i), file=fw)
        print('Inf per point: ',Inf/((bthN-1)*batch_size), file=fw)
        print('Ret+rank  per point: ',RetRank/((bthN-1)*batch_size), file=fw)
        print('per point to report: ',(Inf/32 + RetRank/4)/((bthN-1)*batch_size), file=fw)
        np.save(output_loc,output)
        break

# p.close()

