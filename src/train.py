from config import train_config as config
import tensorflow as tf
import glob
import argparse
import time
import numpy as np
import logging
from utils import _parse_function, _parse_function_dense

try:
    from util import topK
except:
    print('**********************CANNOT IMPORT topK***************************')
    exit()


parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
parser.add_argument("--node", default=0, type=int)
parser.add_argument("--gpu", default='0')
parser.add_argument("--gpu_usage", default=0.11, type=float)
parser.add_argument("--load_epoch", default=0, type=int)
parser.add_argument("--k1", default=25, type=int)
parser.add_argument("--k2", default=10, type=int, help='take top-k2 buckets of accumulated label vectors and assign it to the least occupied')
parser.add_argument("--n_epochs", default=30, type=int)
args = parser.parse_args()

if not args.gpu=='all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

r = int(args.repetition) # which repetition

############################## Test code from here ################################
if args.load_epoch==0:
    lookup = tf.Variable(np.load(config.lookups_loc+'epoch_'+str(args.load_epoch)+'/bucket_order_'+str(r)+'.npy'))
else:
    lookup = tf.Variable(np.load(config.lookups_loc+'epoch_'+str(args.load_epoch)+'/node_'+str(args.node)+'/bucket_order_'+str(r)+'.npy'))

# query_lookup = tf.constant(np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy'))

train_files = glob.glob(config.tfrecord_loc+'*train_'+str(args.node)+'.tfrecords')

dataset = tf.data.TFRecordDataset(train_files)
# dataset = dataset.map(_parse_function, num_parallel_calls=4)
# dataset = dataset.batch(config.batch_size)
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function, batch_size=config.batch_size))
dataset = dataset.prefetch(buffer_size=100)
dataset = dataset.shuffle(buffer_size=100)
# dataset = dataset.repeat(args.n_epochs)
iterator = dataset.make_initializable_iterator()
next_y_idxs, next_y_vals, next_x_idxs, next_x_vals = iterator.get_next()
###############
# x_idxs = tf.stack([next_x_idxs.indices[:,0], tf.gather(query_lookup, next_x_idxs.values)], axis=-1)

x_idxs = tf.stack([next_x_idxs.indices[:,0], next_x_idxs.values], axis=-1)

x_vals = next_x_vals.values
x = tf.SparseTensor(x_idxs, x_vals, [config.batch_size, config.feat_hash_dim])
####
# y_idxs = tf.stack([next_y_idxs.indices[:,0], tf.gather(lookup, next_y_idxs.values-args.node*147939)], axis=-1)
y_idxs = tf.stack([next_y_idxs.indices[:,0], tf.gather(lookup, next_y_idxs.values)], axis=-1)
y_vals = next_y_vals.values
y = tf.SparseTensor(y_idxs, y_vals, [config.batch_size, config.B])
y_ = tf.sparse_tensor_to_dense(y, validate_indices=False)

###############
if args.load_epoch>0:
    params=np.load(config.model_save_loc+'node_'+str(args.node)+'/r_'+str(r)+'_epoch_'+str(args.load_epoch)+'.npz')
    #
    W1_tmp = tf.placeholder(tf.float32, shape=[config.feat_hash_dim, config.hidden_dim])
    b1_tmp = tf.placeholder(tf.float32, shape=[config.hidden_dim])
    W1 = tf.Variable(W1_tmp)
    b1 = tf.Variable(b1_tmp)
    hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2_tmp = tf.placeholder(tf.float32, shape=[config.hidden_dim, config.B])
    b2_tmp = tf.placeholder(tf.float32, shape=[config.B])
    W2 = tf.Variable(W2_tmp)
    b2 = tf.Variable(b2_tmp)
    logits = tf.matmul(hidden_layer,W2)+b2
else:
    W1 = tf.Variable(tf.truncated_normal([config.feat_hash_dim, config.hidden_dim], stddev=0.05, dtype=tf.float32))
    b1 = tf.Variable(tf.truncated_normal([config.hidden_dim], stddev=0.05, dtype=tf.float32))
    hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal([config.hidden_dim, config.B], stddev=0.05, dtype=tf.float32))
    b2 = tf.Variable(tf.truncated_normal([config.B], stddev=0.05, dtype=tf.float32))
    logits = tf.matmul(hidden_layer,W2)+b2


top_buckets = tf.nn.top_k(logits, k=args.k1, sorted=True)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_))
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session(config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=args.gpu_usage)))

if args.load_epoch==0:
    sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer(),
        feed_dict = {
            W1_tmp:params['W1'],
            b1_tmp:params['b1'],
            W2_tmp:params['W2'],
            b2_tmp:params['b2']})
    del params

begin_time = time.time()
total_time = 0
logging.basicConfig(filename = config.logfile+'node_'+str(args.node)+'/logs_'+str(r), level=logging.INFO)
n_check=1000

###############################################
def _parse_function_inp(example_proto): # for reading TFRecords
    features = {"vector": tf.FixedLenFeature(dtype=tf.float32, shape=config.feat_hash_dim)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["vector"]


vec_files = glob.glob(config.tfrecord_loc+str(args.node)+'_vecs'+'.tfrecords')

dataset_vecs = tf.data.TFRecordDataset(vec_files)
dataset_vecs = dataset_vecs.apply(tf.contrib.data.map_and_batch(
    map_func=_parse_function_inp, batch_size=config.batch_size))
dataset_vecs = dataset_vecs.prefetch(buffer_size=100)
iterator_vecs = dataset_vecs.make_initializable_iterator()
vecs = iterator_vecs.get_next()
hidden_layer_vecs = tf.nn.relu(tf.matmul(vecs,W1)+b1)
logits_vecs = tf.matmul(hidden_layer_vecs,W2)+b2
top_buckets_vecs = tf.nn.top_k(logits_vecs, k=args.k2, sorted=True)[1]

###############################################

for curr_epoch in range(args.load_epoch+1,args.load_epoch+args.n_epochs+1):
    sess.run(iterator.initializer)
    count = 0
    while True:
        try:
            sess.run(train_op)
            count += 1
            if count%n_check==0:
                _, train_loss = sess.run([train_op, loss])
                time_diff = time.time()-begin_time
                total_time += time_diff
                logging.info('finished '+str(count)+' steps. Time elapsed for last '+str(n_check)+' steps: '+str(time_diff)+' s')
                logging.info('train_loss: '+str(train_loss))
                begin_time = time.time()
                count+=1
        except tf.errors.OutOfRangeError:
            break
    logging.info('###################################')
    logging.info('finished epoch '+str(curr_epoch))
    logging.info('total time elapsed so far: '+str(total_time))
    logging.info('###################################')
    if curr_epoch%5==0:
        params = sess.run([W1,b1,W2,b2])
        np.savez_compressed(config.model_save_loc+'node_'+str(args.node)+'/r_'+str(r)+'_epoch_'+str(curr_epoch)+'.npz',
            W1=params[0], 
            b1=params[1], 
            W2=params[2], 
            b2=params[3])
        del params
        #######################################
        begin_time = time.time()
        ###################
        # sess.run(iterator.initializer)
        # aff_mat = np.zeros([config.n_classes,config.B], dtype=np.float16)
        # while True:
        #     try:
        #         # top_buckets_, y_idxs_ = sess.run([top_buckets, next_y_idxs_])
        #         # for j in range(y_idxs_[0].shape[0]):
        #         #     aff_mat[y_idxs_[1][j],top_buckets_[1][y_idxs_[0][j,0]]] += top_buckets_[0][y_idxs_[0][j,0]] # - top_buckets_[0][y_idxs_[0][j,0]][-1]
        #         #######
        #         logits_, y_idxs_ = sess.run([logits, next_y_idxs])
        #         temp = np.where(y_idxs_[0][:,1]==0)[0]
        #         temp = np.concatenate([temp,y_idxs_[1].shape])
        #         for j in range(y_idxs_[2][0]):
        #             aff_mat[y_idxs_[1][temp[j]:temp[j+1]]-args.node*147939] = np.tile(logits_[j].astype(np.float16)/100, (temp[j+1]-temp[j],1))
        #         #######
        #     except tf.errors.OutOfRangeError:
        #         break
        # ###
        # print(time.time()-begin_time)
        # # aff_mat[aff_mat==0] = -float('inf')
        # top_preds = np.zeros([config.n_classes,args.k2], dtype=int)
        # overall_count = 0
        # ###
        # for i in range(aff_mat.shape[0]//config.batch_size):
        #     start_idx = overall_count
        #     end_idx = start_idx+config.batch_size
        #     topK(aff_mat[start_idx:end_idx].astype(np.float32), top_preds[start_idx:end_idx], config.B, config.batch_size, args.k2, 2)
        #     overall_count = end_idx
        # ###
        # if overall_count<config.n_classes:
        #     start_idx = overall_count
        #     end_idx = config.n_classes
        #     topK(aff_mat[start_idx:end_idx].astype(np.float32), top_preds[start_idx:end_idx], config.B, end_idx-start_idx, args.k2, 2)
        #     overall_count = end_idx
        ###
        ##################
        sess.run(iterator_vecs.initializer)
        top_preds = np.zeros([config.n_classes,args.k2], dtype=int)
        start_idx = 0
        while True:
            try:
                top_preds[start_idx:start_idx+config.batch_size] = sess.run(top_buckets_vecs)
                start_idx += config.batch_size
            except tf.errors.OutOfRangeError:
                break
        ##
        print(time.time()-begin_time)
        #####################################
        counts = np.zeros(config.B+1, dtype=int)
        bucket_order = np.zeros(config.n_classes, dtype=int)
        for i in range(config.n_classes):
            bucket = top_preds[i, np.argmin(counts[top_preds[i]+1])] 
            bucket_order[i] = bucket
            counts[bucket+1] += 1
        ###
        nothing = sess.run(tf.assign(lookup,bucket_order))
        ###
        counts = np.cumsum(counts)
        rolling_counts = np.zeros(config.B, dtype=int)
        class_order = np.zeros(config.n_classes,dtype=int)
        for i in range(config.n_classes):
            temp = bucket_order[i]
            class_order[counts[temp]+rolling_counts[temp]] = i
            rolling_counts[temp] += 1
        ###
        folder_path = config.lookups_loc+'epoch_'+str(curr_epoch)+'/node_'+str(args.node)
        # if not os.path.isdir(folder_path):
        #     os.system('mkdir '+folder_path)
        np.save(folder_path+'/class_order_'+str(r)+'.npy', class_order)
        np.save(folder_path+'/counts_'+str(r)+'.npy', counts)
        np.save(folder_path+'/bucket_order_'+str(r)+'.npy', bucket_order)
        ################
        begin_time = time.time()


