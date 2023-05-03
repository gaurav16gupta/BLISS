import numpy as np
import h5py
from utils import *
from config import config

# add SIFT and other data as well

def getTraindata(dataname):
    metric = config.DATASET[dataname]['metric']
    datapath = '../../data/{}/'.format(dataname)
    trainpath = datapath + 'train.npy'
    gtpath = datapath + 'groundTruth.npy'

    if os.path.exists(trainpath) and os.path.exists(gtpath):     #check file size as well   
        print ("GT already there")
    else:
        #load the full data and get fraction
        fulldata = getFulldata(dataname, datapath)
        N = fulldata.shape[0]
        if N>10**6:
            # pick = np.random.choice(N, np.clip(N//100, 10**4, 10**6), replace=False) # fix seed
            np.random.seed(0)
            pick = np.random.choice(N, 10**6, replace=False) # fix seed
            data_train = fulldata[pick,:]
        else:
            data_train = fulldata
        # np.random.shuffle(data_train)
        gt = getTrueNNS(data_train, metric, 100)
        np.save(gtpath, gt)
        np.save(trainpath, data_train)
        del fulldata

def getFulldata(dataname, datapath):
    if dataname == 'glove':
        if os.path.exists(datapath+'fulldata.dat'):
            dt = config.DATASET[dataname]['dt'] 
            N = config.DATASET[dataname]['N']
            d = config.DATASET[dataname]['d']
            return np.array(np.memmap(datapath+'fulldata.dat', dtype=dt, mode='c', shape=(N,d)))
        else:
            data = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('train'))
            norms = np.linalg.norm(data,axis=1)
            savememmap(datapath+'fulldata.dat', data)
            np.save(datapath+'norms.npy', norms)
            return data
    if dataname == 'sift':
        if os.path.exists(datapath+'fulldata.dat'):
            dt = config.DATASET[dataname]['dt'] 
            N = config.DATASET[dataname]['N']
            d = config.DATASET[dataname]['d']
            return np.array(np.memmap(datapath+'fulldata.dat', dtype=dt, mode='c', shape=(N,d)))
        else:
            data = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('train'))
            norms = np.linalg.norm(data,axis=1)
            savememmap(datapath+'fulldata.dat', data)
            np.save(datapath+'norms.npy', norms)
            return data

def getQueries(dataname):
    datapath = '../../data/{}/'.format(dataname)
    if dataname == 'glove':
        if os.path.exists(datapath+'queries.npy') and os.path.exists(datapath+ 'neighbors100.npy'): 
            queries = np.load(datapath+'queries.npy')
            neighbors100 = np.load(datapath+ 'neighbors100.npy')
        else:
            queries = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('test'))
            neighbors100 = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('neighbors'))
            np.save(datapath+'queries.npy', queries)
            np.save(datapath+ 'neighbors100.npy', neighbors100)
        return [queries, neighbors100]

    if dataname == 'sift':
        if os.path.exists(datapath+'queries.npy') and os.path.exists(datapath+ 'neighbors100.npy'): 
            queries = np.load(datapath+'queries.npy')
            neighbors100 = np.load(datapath+ 'neighbors100.npy')
        else:
            queries = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('test'))
            neighbors100 = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('neighbors'))
            np.save(datapath+'queries.npy', queries)
            np.save(datapath+ 'neighbors100.npy', neighbors100)
        return [queries, neighbors100]

# if dataname == 'deep-1b':
#     import subprocess
#     import os
#     yadiskLink = "https://yadi.sk/d/11eDCm7Dsn9GA"

#     # download base files
#     for i in range(0,4):
#         command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
#                 + yadiskLink + '&path=/base/base_' + str(i).zfill(2) + '"'
        
#         process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
#         (out, err) = process.communicate()
#         out = out.decode()
#         wgetLink = out.split(',')[0][:]
#         wgetCommand = 'wget ' + wgetLink + ' -O base_' + str(i).zfill(2)
#         os.system(wgetCommand.split('{"href":')[0]+ wgetCommand.split('{"href":')[1])

#         print ("Downloading base chunk " + str(i).zfill(2) + ' ...')
#         #process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
#         #process.stdin.write('e')
#         #process.wait()

#     #curate
#     #convert
#     #split
#     #get groundtruth

#     DATASET = {'name':'deep-1b','N':10**9, 'd':96, 'metric': 'ip', 'dt':'float32', 
#                     'fullpath':'../../data/deep-1b/fulldata.dat', 'trainpath':'../../data/deep-1b/traindata.dat'}


