import numpy as np
import argparse
import os, sys
sys.path.append('../indices/')
import pdb
import argparse
from utils import *
from train import trainIndex
from index import Index
from config import config

parser = argparse.ArgumentParser()
parser.add_argument("--index", default='glove_epc20_K2_B4096_R4', type=str)
parser.add_argument("--gpu", default='0', type=str)
parser.add_argument("--memmap", default=False, type=bool)
args = parser.parse_args()

datasetName = args.index.split('_')[0]  
n_epochs = int(args.index.split('_')[1].split('epc')[1]) 
K = int(args.index.split('_')[2].split('K')[1])  
B = int(args.index.split('_')[3].split('B')[1])
R = int(args.index.split('_')[4].split('R')[1])
feat_dim =  config.DATASET[datasetName]['d']
N = config.DATASET[datasetName]['N'] 
metric = config.DATASET[datasetName]['metric'] 
dtype = config.DATASET[datasetName]['dt'] 

if not os.path.exists("../logs/{}".format(datasetName)):  
    os.makedirs("../logs/{}".format(datasetName))

lookups_loc  = "../indices/{}/".format(datasetName)
train_data_loc = "../../data/{}/".format(datasetName)
model_save_loc = "../indices/{}/".format(datasetName)
batch_size = 5000
hidden_dim = 512
logfile = "../logs/{}/".format(datasetName)
gpu = 0
gpu_usage =0.9
load_epoch = 0
t1 = time.time()
for r in range(R):
    trainIndex(lookups_loc, train_data_loc, datasetName, model_save_loc, batch_size, B, feat_dim, hidden_dim, logfile,
                    r, gpu, gpu_usage, load_epoch, K, n_epochs)

print ("Training finished in: ",time.time()-t1, " sec")

