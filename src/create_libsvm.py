import numpy as np
import tensorflow as tf
from utils import input_example

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'

# W0_tmp = tf.placeholder(tf.float16, shape=[4096, 1250000])
# W0 = tf.Variable(W0_tmp)

# a = tf.cast(tf.square(W0),tf.float32)
# b = tf.reduce_sum(a,axis=0)

# sess.run(a[:,0])

train_x = np.load('../data/yfcc/train_queries.npy')

batch_size = 10

x = tf.placeholder(shape=[None, 4096], dtype=tf.float16)

with tf.device('/gpu:2'):
    W0_tmp = tf.placeholder(tf.float16, shape=[4096, 1250000])
    W0 = tf.Variable(W0_tmp)
    W0_norm = tf.square(tf.norm(W0,axis=0))
    sim_0 = 2*tf.matmul(x,W0) - W0_norm
    topk_0 = tf.nn.top_k(sim_0, k=100, sorted=True)

with tf.device('/gpu:5'):
    W1_tmp = tf.placeholder(tf.float16, shape=[4096, 1250000])
    W1 = tf.Variable(W1_tmp)
    W1_norm = tf.square(tf.norm(W1,axis=0))
    sim_1 = 2*tf.matmul(x,W1) - W1_norm
    topk_1 = tf.nn.top_k(sim_1, k=100, sorted=True)

with tf.device('/gpu:6'):
    W2_tmp = tf.placeholder(tf.float16, shape=[4096, 1250000])
    W2 = tf.Variable(W2_tmp)
    W2_norm = tf.square(tf.norm(W2,axis=0))
    sim_2 = 2*tf.matmul(x,W2) - W2_norm
    topk_2 = tf.nn.top_k(sim_2, k=100, sorted=True)

with tf.device('/gpu:7'):
    W3_tmp = tf.placeholder(tf.float16, shape=[4096, 1250000])
    W3 = tf.Variable(W3_tmp)
    W3_norm = tf.square(tf.norm(W3,axis=0))
    sim_3 = 2*tf.matmul(x,W3) - W3_norm
    topk_3 = tf.nn.top_k(sim_3, k=100, sorted=True)

sess = tf.Session(config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))

sess.run(tf.global_variables_initializer(),
        feed_dict = {
            W0_tmp:np.load('../data/yfcc/node_0.npy').T,
            W1_tmp:np.load('../data/yfcc/node_1.npy').T,
            W2_tmp:np.load('../data/yfcc/node_2.npy').T,
            W3_tmp:np.load('../data/yfcc/node_3.npy').T})

fw0 = tf.python_io.TFRecordWriter('../data/yfcc/tfrecords/train_0.tfrecords')
fw1 = tf.python_io.TFRecordWriter('../data/yfcc/tfrecords/train_1.tfrecords')
fw2 = tf.python_io.TFRecordWriter('../data/yfcc/tfrecords/train_2.tfrecords')
fw3 = tf.python_io.TFRecordWriter('../data/yfcc/tfrecords/train_3.tfrecords')


x_idxs = np.arange(4096)

for i in range(train_x.shape[0]//batch_size):
    start_idx = i*batch_size
    end_idx = start_idx+batch_size
    x_batch = train_x[start_idx:end_idx]
    # topk_0_, topk_1_, topk_2_, topk_3_ = sess.run([topk_0,topk_1,topk_2,topk_3], feed_dict={x:x_batch})
    topk_0_ = sess.run(topk_0, feed_dict={x:x_batch})
    for j in range(batch_size):
        x_vals = x_batch[j]
        ############################
        y_idxs = topk_0_[1][j]
        y_vals = np.ones(len(y_idxs)) # topk_0_[0][j]
        tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
        fw0.write(tf_example.SerializeToString())
        ############################
        # y_idxs = topk_1_[1][j]
        # y_vals = np.ones(len(y_idxs)) # topk_1_[0][j]
        # tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
        # fw1.write(tf_example.SerializeToString())
        # ############################
        # y_idxs = topk_2_[1][j]
        # y_vals = np.ones(len(y_idxs)) # topk_2_[0][j]
        # tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
        # fw2.write(tf_example.SerializeToString())
        # ############################
        # y_idxs = topk_3_[1][j]
        # y_vals = np.ones(len(y_idxs)) # topk_3_[0][j]
        # tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
        # fw3.write(tf_example.SerializeToString())
    ###
    if i==99000:
        break

################ TEST DATA #################
test_x = np.load('../data/yfcc/test_queries.npy')
batch_size = 100

test_scores = np.zeros([10000,800]).astype(np.float16)
test_idxs = np.zeros([10000,800], dtype=np.int32)

for i in range(test_x.shape[0]//batch_size):
    start_idx = i*batch_size
    end_idx = start_idx+batch_size
    x_batch = test_x[start_idx:end_idx]
    topk_0_, topk_1_, topk_2_, topk_3_ = sess.run([topk_0,topk_1,topk_2,topk_3], feed_dict={x:x_batch})
    # topk_0_, topk_1_ = sess.run([topk_0,topk_1], feed_dict={x:x_batch})
    ##
    test_scores[start_idx:end_idx,:100] = topk_0_[0]
    test_idxs[start_idx:end_idx,:100] = topk_0_[1]
    ##
    test_scores[start_idx:end_idx,100:200] = topk_1_[0]
    test_idxs[start_idx:end_idx,100:200] = topk_1_[1] + 1250000
    ##
    test_scores[start_idx:end_idx,200:300] = topk_2_[0]
    test_idxs[start_idx:end_idx,200:300] = topk_2_[1] + 2500000
    ##
    test_scores[start_idx:end_idx,300:400] = topk_3_[0]
    test_idxs[start_idx:end_idx,300:400] = topk_3_[1] + 3750000
    ##
    # test_scores[start_idx:end_idx,400:500] = topk_0_[0]
    # test_idxs[start_idx:end_idx,400:500] = topk_0_[1] + 5000000
    # ##
    # test_scores[start_idx:end_idx,500:600] = topk_1_[0]
    # test_idxs[start_idx:end_idx,500:600] = topk_1_[1] + 6250000
    # ##
    # test_scores[start_idx:end_idx,600:700] = topk_2_[0]
    # test_idxs[start_idx:end_idx,600:700] = topk_2_[1] + 7500000
    # ##
    # test_scores[start_idx:end_idx,700:] = topk_3_[0]
    # test_idxs[start_idx:end_idx,700:] = topk_3_[1] + 8750000


sorted_nns = np.argsort(-test_scores,axis=1)[:,:100]
fw = tf.python_io.TFRecordWriter('../data/yfcc/tfrecords/test.tfrecords')
x_idxs = np.arange(4096)

for i in range(10000):
    x_vals = test_x[i]
    y_idxs = test_idxs[i,sorted_nns[i]]
    y_vals = test_scores[i,sorted_nns[i]]
    tf_example = input_example(y_idxs, y_vals, x_idxs, x_vals)
    fw.write(tf_example.SerializeToString())

