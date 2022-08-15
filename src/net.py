import tensorflow as tf
import numpy as np

class MyModule(tf.Module):
  def __init__(self, R):
    self.R = R
    self.W1 = [None for r in range(R)]
    self.b1 = [None for r in range(R)]
    self.hidden_layer = [None for r in range(R)]
    self.W2 = [None for r in range(R)]
    self.b2 = [None for r in range(R)]
    self.logits = [None for r in range(R)]
    self.top_buckets = [None for r in range(R)]

  def load(self,paths):
    params = [np.load(path) for path in paths]
    self.W1 = [tf.constant(params[r]['W1']) for r in range(self.R)]
    self.b1 = [tf.constant(params[r]['b1']) for r in range(self.R)]
    self.W2 = [tf.constant(params[r]['W2']) for r in range(self.R)]
    self.b2 = [tf.constant(params[r]['b2']) for r in range(self.R)]

  @tf.function
  def __call__(self, x, topk):
    for r in range(self.R):
        self.hidden_layer[r] = tf.nn.relu(tf.matmul(x, self.W1[r])+self.b1[r])
        self.logits[r] = tf.matmul(self.hidden_layer[r],self.W2[r])+self.b2[r]
        self.top_buckets[r] = tf.nn.top_k(self.logits[r], k=topk, sorted=False)[1]
    return self.top_buckets