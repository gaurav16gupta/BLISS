import tensorflow as tf
import argparse
import time
import os
import numpy as np
import logging
from dataPrepare import *
from utils import *
import pdb

def trainIndex(lookups_loc, train_data_loc, datasetName, model_save_loc, batch_size, B, vec_dim, hidden_dim, logfile,
                    r, gpu, gpu_usage, load_epoch, k2, n_epochs):

    tf.compat.v1.disable_eager_execution()

    if not gpu=='all':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # get train data
    getTraindata(datasetName) # check if already there, check if it return correct ground truth
    x_train = np.load(train_data_loc+'train.npy')
    y_train = np.load(train_data_loc+'groundTruth.npy')
    N = x_train.shape[0]
    if not os.path.exists(lookups_loc+'epoch_'+str(load_epoch)+'/'):  
        os.makedirs(lookups_loc+'epoch_'+str(load_epoch)+'/')
    create_universal_lookups(r, B, N, lookups_loc+'epoch_'+str(load_epoch)+'/')

    if load_epoch==0:
        lookup = tf.Variable(np.load(lookups_loc+'epoch_'+str(load_epoch)+'/bucket_order_'+str(r)+'.npy')[:N])
    else:
        lookup = tf.Variable(np.load(lookups_loc+'epoch_'+str(load_epoch)+'/bucket_order_'+str(r)+'.npy')[:N])

    temp = tf.constant(np.arange(batch_size*100)//100)

    x = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, vec_dim])
    _y = tf.compat.v1.placeholder(tf.int64, shape=[batch_size*100])
    y_idxs = tf.stack([temp, tf.gather(lookup, _y)], axis=-1)
    y_vals = tf.ones_like(y_idxs[:,0], dtype=tf.float32)
    y = tf.compat.v1.SparseTensor(y_idxs, y_vals, [batch_size, B])
    y_ = tf.compat.v1.sparse_tensor_to_dense(y, validate_indices=False)

    ###############
    if load_epoch>0:
        params=np.load(model_save_loc+'/r_'+str(r)+'_epoch_'+str(load_epoch)+'.npz')
        #
        W1_tmp = tf.compat.v1.placeholder(tf.float32, shape=[vec_dim, hidden_dim])
        b1_tmp = tf.compat.v1.placeholder(tf.float32, shape=[hidden_dim])
        W1 = tf.Variable(W1_tmp)
        b1 = tf.Variable(b1_tmp)
        # hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
        hidden_layer = tf.nn.relu(tf.matmul(x,W1)+b1)
        #
        W2_tmp = tf.compat.v1.placeholder(tf.float32, shape=[hidden_dim, B])
        b2_tmp = tf.compat.v1.placeholder(tf.float32, shape=[B])
        W2 = tf.Variable(W2_tmp)
        b2 = tf.Variable(b2_tmp)
        logits = tf.matmul(hidden_layer,W2)+b2
    else:
        W1 = tf.Variable(tf.compat.v1.truncated_normal([vec_dim, hidden_dim], stddev=0.05, dtype=tf.float32))
        b1 = tf.Variable(tf.compat.v1.truncated_normal([hidden_dim], stddev=0.05, dtype=tf.float32))
        # hidden_layer = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
        hidden_layer = tf.nn.relu(tf.matmul(x,W1)+b1)
        #
        W2 = tf.Variable(tf.compat.v1.truncated_normal([hidden_dim, B], stddev=0.05, dtype=tf.float32))
        b2 = tf.Variable(tf.compat.v1.truncated_normal([B], stddev=0.05, dtype=tf.float32))
        logits = tf.matmul(hidden_layer,W2)+b2


    top_buckets = tf.nn.top_k(logits, k=k2, sorted=True)[1]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_))
    train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(
                            allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_usage)))

    if load_epoch==0:
        sess.run(tf.compat.v1.global_variables_initializer())
    else:
        sess.run(tf.compat.v1.global_variables_initializer(),
            feed_dict = {
                W1_tmp:params['W1'],
                b1_tmp:params['b1'],
                W2_tmp:params['W2'],
                b2_tmp:params['b2']})
        ##
        del params

    begin_time = time.time()
    total_time = 0
    logging.basicConfig(filename = logfile+'/logs_'+str(r), level=logging.INFO)
    n_check=1000


    n_steps_per_epoch = N//batch_size

    for curr_epoch in range(load_epoch+1,load_epoch+n_epochs+1):
        count = 0
        
        for j in range(n_steps_per_epoch):
            start_idx = j*batch_size
            end_idx = start_idx+batch_size
            # pdb.set_trace()
            try:
                sess.run(train_op, feed_dict={x:x_train[start_idx:end_idx], _y:y_train[start_idx:end_idx].reshape([-1])})
            except:
                pdb.set_trace()
            count += 1
            if count%n_check==0:
                _, train_loss = sess.run([train_op, loss], feed_dict={x:x_train[start_idx:end_idx], _y:y_train[start_idx:end_idx].reshape([-1])})
                time_diff = time.time()-begin_time
                total_time += time_diff
                logging.info('finished '+str(count)+' steps. Time elapsed for last '+str(n_check)+' steps: '+str(time_diff)+' s')
                logging.info('train_loss: '+str(train_loss))
                begin_time = time.time()
                count+=1
        ############################################
        logging.info('###################################')
        logging.info('finished epoch '+str(curr_epoch))
        logging.info('total time elapsed so far: '+str(total_time))
        logging.info('###################################')
        if curr_epoch%5==0:
            params = sess.run([W1,b1,W2,b2])
            np.savez_compressed(model_save_loc+'/r_'+str(r)+'_epoch_'+str(curr_epoch)+'.npz',
                W1=params[0], 
                b1=params[1], 
                W2=params[2], 
                b2=params[3])
            del params
            #######################################
            begin_time = time.time()
            
            top_preds = np.zeros([N,k2], dtype=int)
            start_idx = 0
            for i in range(x_train.shape[0]//batch_size):
                top_preds[start_idx:start_idx+batch_size] = sess.run(top_buckets, feed_dict={x:x_train[start_idx:start_idx+batch_size]})
                start_idx += batch_size
            ##
            # top_preds[start_idx:] = sess.run(top_buckets, feed_dict={x:x_train[start_idx:]})
            ##
            print(time.time()-begin_time)
            ##################################### 
            counts = np.zeros(B+1, dtype=int)
            bucket_order = np.zeros(N, dtype=int)
            for i in range(N):
                bucket = top_preds[i, np.argmin(counts[top_preds[i]+1])] 
                bucket_order[i] = bucket
                counts[bucket+1] += 1
            ###
            nothing = sess.run(tf.compat.v1.assign(lookup,bucket_order))
            ###
            counts = np.cumsum(counts)
            rolling_counts = np.zeros(B, dtype=int)
            class_order = np.zeros(N,dtype=int)
            for i in range(N):
                temp = bucket_order[i]
                class_order[counts[temp]+rolling_counts[temp]] = i
                rolling_counts[temp] += 1
            
            ###
            # folder_path = lookups_loc+'epoch_'+str(curr_epoch)
            # if not os.path.isdir(folder_path):
            #     os.makedirs(folder_path)
            # np.save(folder_path+'/class_order_'+str(r)+'.npy', class_order)
            # np.save(folder_path+'/counts_'+str(r)+'.npy', counts)
            # np.save(folder_path+'/bucket_order_'+str(r)+'.npy', bucket_order)
            ################
            begin_time = time.time()


