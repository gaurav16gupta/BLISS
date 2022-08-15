# Todo: 
# bucket reorder: 1) during insertion keep a count, for each bucket how many times each other bucket got featured. Store this count and form a similarty matrix
#   near diagonalise this matrix, 2) or reorder the 2nd Weight matrix based on nearness of rows
# Data reorder on R=1 _ done
# Instead of keeping N size array, use a Bloom filter with very cheap hash calculation-
#   Insert the point if it is not there and retrive the point if it was there already. This can work for Th=2, For more use count BF
# 32 parallel query, 32 cpu code
# try huge pages


import tensorflow as tf
tf.get_logger().setLevel('WARNING')
import time
import numpy as np
import logging
import os, sys
import pdb
sys.path.append('InvertedIndex/')
import scoreAgg
from net import MyInferenceModule
import matplotlib.pyplot as plt

# import argparse
# import json
# import glob
# from multiprocessing import Pool

class BLISS():
    def __init__(self,params):
        [self.datasetName, self.epc, self.K, self.B, self.R, self.feat_dim, self.N, self.batch_size, 
        self.hidden_dim, self.metric, self.dtype, self.logfile, self.rerank] = params

    # def train(self, traindata):
    # def index(self, indexdata):

    def load_index(self, model_loc, lookups_loc, data_loc, topk, mf, reorder="False"):
        ############################## load lookups ################################
        self.Model = MyInferenceModule(self.R)
        self.Model.load([model_loc+'_r'+str(r)+'.npz' for r in range(self.R)])
        print ("model loaded")

        self.topk = topk
        self.mf = mf
        self.inv_lookup = np.zeros((self.N)*self.R, dtype=np.int32)
        self.counts = np.zeros((self.B+1)*self.R, dtype=np.int32)
        
        if (reorder=="True"):
            self.inv_lookup = np.array(np.memmap(lookups_loc+'lookupReordered_R'+str(self.R)+'.dat', dtype='int32', mode='r', shape=(self.N*self.R)))
            self.counts = np.array(np.memmap(lookups_loc+'counts_R'+str(self.R)+'.dat', dtype='int32', mode='r', shape=((self.B+1)*self.R)))
        else:
            self.inv_lookup = np.array(np.memmap(lookups_loc+'lookup_R'+str(self.R)+'new.dat', dtype='uint32', mode='r', shape=(self.N*self.R)))
            self.counts = np.array(np.memmap(lookups_loc+'counts_R'+str(self.R)+'new.dat', dtype='uint32', mode='r', shape=((self.B+1)*self.R)))
            # for r in range(self.R):
            #     mm = np.memmap(lookups_loc+'lookup_r'+str(r)+'.dat', dtype='int32', mode='r', shape=(self.N+self.B+1))
            #     self.inv_lookup[r*(self.N): (r +1)*(self.N)] = np.array(mm[:self.N])
            #     self.counts[r*(self.B+1) : (r +1)*(self.B+1)] = np.cumsum(np.array(mm[self.N:]))
        self.inv_lookup = np.ascontiguousarray(self.inv_lookup, dtype=np.int32) 
        self.counts = np.ascontiguousarray(self.counts, dtype=np.int32)
        print ("Deserialized")
        self.fastIv = scoreAgg.PyFastIV(self.R, self.N, (self.B+1), mf, topk, self.inv_lookup, self.counts)
        print ("cppDeserialized")
        
        # dataset = {}
        # norms = {}
        if self.rerank:
            if (reorder=="True"):
                self.dataset= (np.memmap(data_loc +'reordereddata.dat', dtype=self.dtype, mode='r', shape=(self.N,self.feat_dim)))
                if self.metric=="L2":
                    self.norms = np.load(datanorm_loc)
            else:
                self.dataset= (np.memmap(data_loc +'fulldata.dat', dtype=self.dtype, mode='r', shape=(self.N,self.feat_dim)))
                if self.metric=="L2":
                    self.norms = np.load(datanorm_loc)
            
            print ("dense vectors leaded")
        self.fw = open(self.logfile, 'a', encoding='utf-8') # log file

    def getCandidateTrivial(self, top_buckets):
        candidates = []
        top_buckets  = top_buckets.astype('int64')
        for r in range(self.R):
            for k in range(self.topk):
                st = self.counts[ (self.B+1)*r + top_buckets[r,k]] + (self.N)*r
                end = self.counts[(self.B+1)*r + top_buckets[r,k]+1] + (self.N)*r
                # pdb.set_trace()
                candidates.append(self.inv_lookup[st:end])
        candidates = np.concatenate(candidates)
        # pdb.set_trace()
        vals, counts = np.unique(candidates, return_counts=True)
        return vals[counts>=self.mf]

    def query(self, queries):
        os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
        bthN = 0
        begin_time = time.time()
        Inf = 0
        RetRank = 0
        score_sum = [0.0,0.0,0.0]
        output = -1* np.ones([queries.shape[0],10])
        queries = tf.data.Dataset.from_tensor_slices(queries)
        queries = queries.batch(batch_size = self.batch_size)
        iterator = iter(queries)
        maxsize = int(2*self.N*self.R*self.topk/(self.B*(self.mf)**2))
        # maxsize = 65536
        avgCand = 0
        uqCand = 0
        while True:
            try:
                x_batch = iterator.get_next()
                x_batch = tf.cast(x_batch, tf.float32) # we don't need to do that, as some data has lesser precision
                t1 = time.time()
                top_buckets_ = self.Model(x_batch, self.topk) # should give topk bucket IDs, [R,2, batch_size,topkvals]
                top_buckets_ = np.array(top_buckets_)
                top_buckets_ = np.transpose(top_buckets_, (2,0,1,3)) # bring batch_size (index 2) ahead, [batch_size,R,2,topk]
                len_cands = np.zeros(top_buckets_.shape[0])
                t2 = time.time()
                Inf+= (t2-t1)
                cppinf=0
                
                for i in range(top_buckets_.shape[0]):
                    # candidates = np.zeros(maxsize, dtype='int32') # does this init takes time?
                    # candidates = np.ascontiguousarray(candidates, dtype=np.int32) 
                    # ti = time.time()
                    # candSize = self.fastIv.FC(np.ascontiguousarray(top_buckets_[i,:,1,:], dtype=np.int32).reshape(-1), maxsize, candidates)
                    # tj = time.time()
                    # cppinf+=tj-ti
                    # candidates = candidates[:candSize]

                    candidates = self.getCandidateTrivial(top_buckets_[i,:,1,:])
                    candSize = len(candidates)

                    # ta = time.time()
                    # # candidates = np.sort(candidates) # to make this faster
                    # print (time.time()-ta)
                    avgCand += candSize
                    uqCand += len(candidates)

                    if self.rerank:
                        # for part in range(Parts): 
                        #     start_lbl = (args.node*4 + part)*block_size 
                        if self.metric == "ip":
                            dists = np.dot(self.dataset[candidates],x_batch[i]) # or L2 dist
                        if self.metric == "L2":
                            dists = 2*np.dot(self.dataset[candidates],x_batch[i]) -norms[candidates]
                        if len(dists)<=10:
                            output[bthN*self.batch_size + i, :len(dists)] = candidates
                        else:
                            top_cands = np.argpartition(dists, -10)[-10:]
                            output[bthN*self.batch_size + i, :] = candidates[top_cands] 
                    # distsall = np.dot(self.dataset,x_batch[i])
                    # top_candsall = np.argpartition(distsall, -10)[-10:]

                    # print (np.intersect1d(top_candsall,candidates[top_cands]))
                    # pdb.set_trace()
                    # t3 = time.time()
                    # print("distcomp ",t3-t2)
                        # score_sum[0] += len(np.intersect1d(candidates, neighbors[i,:10]))/10
                        # len_cands[i] += len(candidates)
                    # tc = time.time()
                    # print('Rank: ',(tc-tb))
                    # score_sum[1] += np.sum(len_cands)
                    # pdb.set_trace()
                    # shortlist = p.map(process_scores, top_buckets_) # parallel over points in the batch, 2 D array
                t3 = time.time()
                RetRank+= t3-t2
                bthN+=1
                print (bthN)
                print ("cppinf: ", cppinf/self.batch_size)
                print('Ret+rank per point: ',RetRank/(bthN*self.batch_size))
                print ("avgCand: ", avgCand/(bthN*self.batch_size))
                # print ("uqCand: ", uqCand/(bthN*self.batch_size))
            except(tf.errors.OutOfRangeError):
                # print ("node: ", args.node, " topk: ", args.topk, " mf: ", args.mf, file=fw)
                # print('overall Recall for',count,'points:',score_sum[0]/(bthN*config.batch_size + i), file=fw)
                # print('Avg can. size for',count,'points:',score_sum[1]/(bthN*config.batch_size + i), file=fw)
                # print('Inf per point: ',Inf/(bthN*config.batch_size), file=fw)
                # print('Ret+rank  per point: ',RetRank/(bthN*config.batch_size), file=fw)
                # print('per point to report: ',(Inf/32 + RetRank/4)/(bthN*config.batch_size), file=fw)
                # np.save(config.output_loc,output)
                break
        return output

    # def testjointIndex(self, lookups_loc):
    #     codebook = np.load(lookups_loc+ 'cacheEff/'+ 'codebook.npy')
    #     Lookups = np.load(lookups_loc+ 'cacheEff/'+'Lookups.npy')

    #     markers1 = np.load(lookups_loc+ 'cacheEff/'+'Markers1.npy')
    #     markers2 = np.load(lookups_loc+ 'cacheEff/'+'Markers2.npy')

    #     inv_lookup = np.array(np.memmap(lookups_loc+'lookup_R'+str(self.R)+'.dat', dtype='int32', mode='r', shape=(self.N*self.R)))
    #     counts = np.array(np.memmap(lookups_loc+'counts_R'+str(self.R)+'.dat', dtype='int32', mode='r', shape=((self.B+1)*self.R)))
        
    #     for pt in range(1,10):
    #         rs = codebook[pt]//(self.B*self.B)
    #         r1 = int(rs//self.R)
    #         r2 = int(rs%self.R)
    #         b1 = int((codebook[pt]%(self.B*self.B))//self.B)
    #         b2 = int(codebook[pt]%self.B)

    #         code = ((r1*self.R + r2)*self.B*self.B) +(b1*self.B) + b2
    #         print (codebook[pt]==code)

    #         stpt1 = counts[r1*(self.B+1)+b1] + r1*self.N
    #         endpt1 = counts[r1*(self.B+1)+b1+1] + r1*self.N
    #         stpt2 = counts[r2*(self.B+1)+b2] + r2*self.N
    #         endpt2 = counts[r2*(self.B+1)+b2+1] + r2*self.N

    #         a = np.sort(np.intersect1d(inv_lookup[stpt1:endpt1], inv_lookup[stpt2:endpt2]).astype('uint32'))
    #         b = np.sort(Lookups[markers1[pt]: markers2[pt]].astype('uint32'))
    #         print (a.shape, b.shape)
    #         print ("match: ", sum(a==b))

    #     pdb.set_trace()

    # do before data reorder, incomplete
    def bucketReorder(self, lookups_loc):
        # rearrange weigh matrix and inv_lookup and counts
        Newinv_lookup = np.zeros((self.N)*self.R, dtype=np.int32)
        Newcounts = np.zeros((self.B+1)*self.R, dtype=np.int32)
        # model weight similarity matrix, bucket reorder all reps 
        for r,W2 in enumerate(self.Model.W2):
            order = self.MatrixReorder(W2)
            pdb.set_trace()
            W2 = W2[order]
            i =0
            # check again
            for j,b in enumerate(order):
                stpt = self.counts[b]
                endpt =self.counts[b+1]
                sz = self.counts[b+1]- self.counts[b]
                Newcounts[j+1] = self.counts[b]
                Newinv_lookup[i:i+sz] = self.inv_lookup[stpt:endpt]
                i = i+sz
        
    # Index time operation
    def dataReorder(self, lookups_loc):
        Newinv_lookup = np.zeros((self.N)*self.R, dtype=np.int32)
        # Newcounts = np.zeros((self.B+1)*self.R, dtype=np.int32)

        reorderDict = np.argsort(self.inv_lookup[0: self.N])
        print ("got reorderDict")
        for r in range(self.R):
            Newinv_lookup[r*(self.N): (r +1)*(self.N)]=reorderDict[self.inv_lookup[r*(self.N): (r +1)*(self.N)]]
        
        print ("saving now")
        fp = np.memmap(lookups_loc+'lookup_R4.dat', dtype='int32', mode='w+', shape=(self.N*self.R))
        fp[:]= self.inv_lookup[:]
        fp.flush()
        
        fp = np.memmap(lookups_loc+'lookupReordered_R4.dat', dtype='int32', mode='w+', shape=(self.N*self.R))
        fp[:]= Newinv_lookup[:]
        fp.flush()

        fp = np.memmap(lookups_loc+'counts_R4.dat', dtype='int32', mode='w+', shape=((self.B+1)*self.R))
        fp[:]= self.counts[:]
        fp.flush()
        print ("saved")

        print (self.dataset[0,:])
        Newdata = self.dataset[self.inv_lookup[0: self.N],:] # reorder based on R=1, 
        print (Newdata[0,:])
        #
        # Newdata = Newdata*(2**15) # mul. by int16 max range
        # Newdata = Newdata.astype('int16')
        #
        fp = np.memmap( '../../data/'+self.datasetName+'/reordereddata.dat', dtype=self.dtype, mode='w+', shape=(self.N,self.feat_dim))
        fp[:]= Newdata[:]
        fp.flush()
        print ("saved data")
        #key
        fp = np.memmap(lookups_loc+'key.dat', dtype='int32', mode='w+', shape=(self.N))
        fp[:]= self.inv_lookup[0: self.N][:]
        fp.flush()


