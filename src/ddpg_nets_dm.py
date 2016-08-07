import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import layers

def hist_summaries(*args):
  return tf.merge_summary([tf.histogram_summary(t.name,t) for t in args])

def fanin_init(shape,fanin=None):
  fanin = fanin or shape[0]
  v = 1/np.sqrt(fanin)
  return tf.random_uniform(shape,minval=-v,maxval=v)


l1 = 400 # dm 400
l2 = 300 # dm 300

# batch_norm_params = {"decay": 0.99, "center": True, "scale": True, "is_training": True, "trainable": True, "reuse": True}
batch_norm_params = {"center": True, "scale": True, "updates_collections": None, "activation_fn": tf.nn.relu}

def batch_norm(input, is_training=None, scope=None, reuse=False):
  # return tf.cond(tf.equal(is_training, tf.constant(1, dtype=tf.int8)),
  return tf.cond(is_training,
    lambda: layers.batch_norm(input, is_training=True, scope=scope, reuse=reuse, **batch_norm_params),
    lambda: layers.batch_norm(input, is_training=False, scope=scope, reuse=True, **batch_norm_params))

def theta_p(dimO,dimA):
  dimO = dimO[0]
  dimA = dimA[0]
  with tf.variable_scope("theta_p"):
    return [tf.Variable(fanin_init([dimO,l1]),name='1w'),
            tf.Variable(fanin_init([l1],dimO),name='1b'),
            tf.Variable(fanin_init([l1,l2]),name='2w'),
            tf.Variable(fanin_init([l2],l1),name='2b'),
            tf.Variable(tf.random_uniform([l2,dimA],-3e-3,3e-3),name='3w'),
            tf.Variable(tf.random_uniform([dimA],-3e-3,3e-3),name='3b')]
  
def policy(obs,theta,name='policy'):
  with tf.variable_op_scope([obs],name,name):
    h0 = tf.identity(obs,name='h0-obs')
    h1 = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
    h2 = tf.nn.relu( tf.matmul(h1,theta[2]) + theta[3],name='h2')
    h3 = tf.identity(tf.matmul(h2,theta[4]) + theta[5],name='h3')
    action = tf.nn.tanh(h3,name='h4-action')
    # print(action.get_shape())
    summary = hist_summaries(h0,h1,h2,h3,action)
    return action,summary

def policy_norm(obs, theta, is_training=None, reuse=False, name='policy'):
  print("Executing Batch normalization on actor")
  with tf.variable_op_scope([obs],name,name):
    h0 = tf.identity(obs,name='h0-obs')
    # var_collections = [tf.zeros([1,], name="beta")]
    with tf.variable_scope('h1') as scope_h1:
      h1 = batch_norm(tf.matmul(h0,theta[0]), is_training=is_training, scope=scope_h1, reuse=reuse)
    with tf.variable_scope('h2') as scope_h2:
      h2 = batch_norm(tf.matmul(h1,theta[2]), is_training=is_training, scope=scope_h2, reuse=reuse)
    h3 = tf.identity(tf.matmul(h2,theta[4]) + theta[5], name='h3' )
    action = tf.nn.tanh(h3,name='h4-action')
    # print(action.get_shape())
    summary = hist_summaries(h0,h1,h2,h3,action)
    return action,summary

def theta_q(dimO,dimA):
  dimO = dimO[0]
  dimA = dimA[0]
  with tf.variable_scope("theta_q"):
    return [tf.Variable(fanin_init([dimO,l1]),name='1w'),
            tf.Variable(fanin_init([l1],dimO),name='1b'),
            tf.Variable(fanin_init([l1+dimA,l2]),name='2w'),
            tf.Variable(fanin_init([l2],l1+dimA),name='2b'),
            tf.Variable(tf.random_uniform([l2,1],-3e-4,3e-4),name='3w'),
            tf.Variable(tf.random_uniform([1],-3e-4,3e-4),name='3b')]
    
def qfunction(obs,act,theta, name="qfunction"):
  with tf.variable_op_scope([obs,act],name,name):
    h0 = tf.identity(obs,name='h0-obs')
    h0a = tf.identity(act,name='h0-act')
    h1  = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
    h1a = tf.concat(1,[h1,act])
    h2  = tf.nn.relu( tf.matmul(h1a,theta[2]) + theta[3],name='h2')
    qs  = tf.matmul(h2,theta[4]) + theta[5]
    q = tf.squeeze(qs,[1],name='h3-q')
    
    summary = hist_summaries(h0,h0a,h1,h2,q)
    return q,summary

def qfunction_norm(obs, act, theta, is_training=None, reuse=False, name="qfunction"):
  print("Executing Batch normalization on critic")
  with tf.variable_op_scope([obs,act],name,name):
    h0 = tf.identity(obs,name='h0-obs')
    h0a = tf.identity(act,name='h0-act')
    with tf.variable_scope('h1') as scope_h1:
      h1  = batch_norm( tf.matmul(h0,theta[0]), is_training=is_training, scope=scope_h1, reuse=reuse)
    h1a = tf.concat(1,[h1,act])
    h2  = tf.nn.relu( tf.matmul(h1a,theta[2]) + theta[3],name='h2')
    qs  = tf.matmul(h2,theta[4]) + theta[5]
    q = tf.squeeze(qs,[1],name='h3-q')
    
    summary = hist_summaries(h0,h0a,h1,h2,q)
    return q,summary