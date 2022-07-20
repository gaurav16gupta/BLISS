from config import train_config as config
import tensorflow as tf
import glob
import argparse
import time
import numpy as np
import logging
from utils import _parse_function

try:
    from util import topK
except:
    print('**********************CANNOT IMPORT topK***************************')
    exit()


parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
parser.add_argument("--gpu", default=1)
parser.add_argument("--gpu_usage", default=0.10, type=float)
parser.add_argument("--load_epoch", default=5, type=int)
parser.add_argument("--k1", default=25, type=int)
parser.add_argument("--k2", default=25, type=int)
args = parser.parse_args()

if not args.gpu=='all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

r = int(args.repetition) # which repetition

############################## Test code from here ################################
# query_lookup = tf.constant(np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy'))

#############
params = np.load(config.model_save_loc+'r_'+str(r)+'_epoch_'+str(args.load_epoch)+'.npz')
W1_tmp = params['W1']
b1_tmp = params['b1']
W2_tmp = params['W2']
b2_tmp = params['b2']

#############
train_files = glob.glob(config.tfrecord_loc+'*train*.tfrecords')

dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function, batch_size=config.batch_size))
dataset = dataset.prefetch(buffer_size=100)
dataset = dataset.shuffle(buffer_size=100)
iterator = dataset.make_initializable_iterator()
next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()

# x = tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup, next_x_idxs.values)], axis=-1),
#     next_x_vals.values, [config.batch_size, config.feat_hash_dim])

x = tf.SparseTensor(tf.stack([next_x_idxs.indices[:,0], next_x_idxs.values], axis=-1),
    next_x_vals.values, [config.batch_size, config.feat_hash_dim])

###############
W1 = tf.constant(W1_tmp)
b1 = tf.constant(b1_tmp)
hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
#
W2 = tf.constant(W2_tmp)
b2 = tf.constant(b2_tmp)
######
logits = tf.matmul(hidden_layer,W2)+b2
probs = tf.sigmoid(logits)
top_buckets = tf.nn.top_k(logits, k=args.k1, sorted=True)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

################
sess.run(iterator.initializer)

begin_time = time.time()

aff_mat = np.zeros([config.n_classes,config.B], dtype=np.float32)
while True:
    try:
        top_buckets_, y_idxs = sess.run([top_buckets, next_y_idxs])
        for j in range(y_idxs[0].shape[0]):
            aff_mat[y_idxs[1][j],top_buckets_[1][y_idxs[0][j,0]]] += top_buckets_[0][y_idxs[0][j,0]] # - top_buckets_[0][y_idxs[0][j,0]][-1]
    except tf.errors.OutOfRangeError:
        break


aff_mat[aff_mat==0] = -float('inf')

print(time.time()-begin_time)

begin_time = time.time()
top_preds = np.zeros([config.n_classes,args.k2], dtype=int)

overall_count = 0
for i in range(aff_mat.shape[0]//config.batch_size):
    start_idx = overall_count
    end_idx = start_idx+config.batch_size
    topK(aff_mat[start_idx:end_idx], top_preds[start_idx:end_idx], config.B, config.batch_size, args.k2, 2)
    overall_count = end_idx

if overall_count<config.n_classes:
    start_idx = overall_count
    end_idx = config.n_classes
    topK(aff_mat[start_idx:end_idx], top_preds[start_idx:end_idx], config.B, end_idx-start_idx, args.k2, 2)
    overall_count = end_idx

print(time.time()-begin_time)

#############
# counts = np.zeros(config.B+1, dtype=int)
# bucket_order = np.zeros(config.n_classes, dtype=int)

# max_in_buck = (config.n_classes//config.B) * 1.5
# lol = 0

# for i in range(config.n_classes):
#     flag = True
#     for j in range(args.k2):
#         if counts[top_preds[i,j]+1]<max_in_buck:
#             bucket_order[i] = top_preds[i,j]
#             counts[top_preds[i,j]+1] += 1
#             flag = False
#             break
#     if flag:
#         lol += 1

##############

counts = np.zeros(config.B+1, dtype=int)
bucket_order = np.zeros(config.n_classes, dtype=int)

for i in range(config.n_classes):
    bucket = top_preds[i, np.argmin(counts[top_preds[i]+1])] 
    bucket_order[i] = bucket
    counts[bucket+1] += 1

##############

np.max(counts)
np.sum(counts==0)
np.std(counts[1:])
np.histogram(counts,bins=[0,10,20,30,40,50,60,110])


counts = np.cumsum(counts)
rolling_counts = np.zeros(config.B, dtype=int)
class_order = np.zeros(config.n_classes,dtype=int)
for i in range(config.n_classes):
    temp = bucket_order[i]
    class_order[counts[temp]+rolling_counts[temp]] = i
    rolling_counts[temp] += 1

np.save(config.lookups_loc+'class_order_'+str(r)+'.npy', class_order)
np.save(config.lookups_loc+'counts_'+str(r)+'.npy',counts)
np.save(config.lookups_loc+'bucket_order_'+str(r)+'.npy', bucket_order)

