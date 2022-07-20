# post query process
import numpy as np
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--rerank", default=False, type=bool)
parser.add_argument("--reorder", default=False, type=bool)
args = parser.parse_args()

R =1
K = 10
Parts = 4
Nodes = 8
output = -1* np.ones([10000,Parts*Nodes*K])
topk = 15
mf = 1
eval_epoch = 20
block_size = (10**9)//32  

# data load if reranking,  one part is 12GB
dataset = {}
# if args.rerank:
#     for node in range(Nodes):
#         for part in range(Parts):
#             datapath = config.eval_data_loc + 'yandex-1b_'+str(node*4 + part)+'.h5'
#             dataset[node*4 + part]= np.array(h5py.File(datapath, 'r').get('train'))

for node in range(Nodes):
    output_loc = '../logs/yandex/b_65536/node_'+str(node)+'/1BR_'+str(R)+'_topk_'+str(topk)+'_mf_'+str(mf)+'_epc_'+str(eval_epoch)+'.npy'
    output[:,node*Parts*K:(node+1)*Parts*K] = np.load(output_loc)
    print (output_loc)

neighborspath =  '../../data/yandex/neighbors100.npy'
neighbors = np.load(neighborspath)

# if args.reorder: 
#     for node in range(Nodes):
#         for part in range(Parts):
#             key = (np.load('../lookups/yandex/b_65536/reordered/node_'+str(node)+'/reorderKey_'+str(part)+'.npy'))
#             for i in range(output.shape[0]):
#                 try:
#                     output[i, (node*Parts+ part)*K:(node*Parts+ part+1)*K] = key[output[i, (node*Parts+ part)*K:(node*Parts+ part+1)*K].astype(int)- (node*4 + part)*block_size]
#                 except:
#                     pdb.set_trace()

score_sum = 0
for i in range(output.shape[0]):
    score_sum += len(np.intersect1d(output[i, :], neighbors[i,:10]))/10

Recall = score_sum/output.shape[0]
print (Recall)