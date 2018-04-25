# Deepnodal interface for TensorFlow
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, l1_regularizer, l2_regularizer
from tensorflow import name_scope, variable_scope
from tensorflow.contrib.framework import arg_scope
from deepnodal.calls.tf_extended import *

#-------------------------------------------------------------------------------
try:
  TF_AUTO_REUSE = tf.AUTO_REUSE
except AttributeError:
  TF_AUTO_REUSE = False

#-------------------------------------------------------------------------------
# Creation dictionary

creation_dict = {'identity': tf.identity,
                 'device': tf.device,
                 'add': tf.add,
                 'add_ewise': tf.add_n,
                 'subtract': tf.subtract,
                 'multiply': tf.multiply,
                 'sum': tf.reduce_sum,
                 'con': tf.concat,
                 'mean': tf.reduce_mean,
                 'dense': tf.layers.dense,
                 'conv2d': tf.layers.conv2d,
                 'pool2d': {'max':tf.layers.max_pooling2d,
                            'avg':tf.layers.average_pooling2d},
                 'flatten': tf.layers.flatten,
                 'batch_norm': tf.layers.batch_normalization,
                 'lresp_norm': tf.nn.local_response_normalization,
                 'dropout': tf.layers.dropout,
                 'l1_reg': l1_regularizer,
                 'l2_reg': l2_regularizer,
                 'max_norm': tf_max_norm_regularizer,
                 'relu': tf.nn.relu,
                 'elu': tf.nn.elu,
                 'softmax': tf.nn.softmax,
                 'var': tf.Variable,
                 'tensor': tf.placeholder,
                 'vsi': tf.variance_scaling_initializer,
                 'zoi': tf.zeros_initializer,
                 'lvi': tf.local_variables_initializer,
                 'gvi': tf.global_variables_initializer,
                 'sgd': tf.train.GradientDescentOptimizer,
                 'mom': tf.train.MomentumOptimizer,
                 'adagrad': tf.train.AdagradOptimizer,
                 'rmsprop': tf.train.RMSPropOptimizer,
                 'adam': tf.train.AdamOptimizer,
                 'mean_squared_error': tf.losses.mean_squared_error,
                 'in_top_k_error': tf_in_top_k_error,
                 'mce': tf_mean_cross_entropy,
                 'logger': tf.summary.FileWriter,
                 'saver': tf.train.Saver,
                 'defaults': tf.get_default_graph,
                 'session': tf.Session}

#-------------------------------------------------------------------------------
def Creation(*args):
  if not(len(args)): return None
  creation = creation_dict
  for arg in args:
    creation = arg if type(arg) is not str else creation[arg.lower()]
  return creation

#-------------------------------------------------------------------------------
# Dtype dictionary

dtype_dict = {None: None,
              'bool': tf.bool,
              'int32': tf.int32,
              'int64': tf.int64,
              'float32': tf.float32,
              'float64': tf.float64,
              'tensor': tf.Tensor}

#-------------------------------------------------------------------------------
def Dtype(arg):
  return dtype_dict[arg]

#-------------------------------------------------------------------------------
def Device(spec):
  return creation_dict['device'](spec)

#-------------------------------------------------------------------------------
# Scope dictionary

scope_dict = {'name': name_scope,
              'var': variable_scope,
              'arg': arg_scope}

#-------------------------------------------------------------------------------
def Scope(spec, *args, **kwds):
  return scope_dict[spec](*args, **kwds)

#-------------------------------------------------------------------------------
# Keys dictionary
keys_dict = {'reg': tf.GraphKeys.REGULARIZATION_LOSSES}

#-------------------------------------------------------------------------------
def Keys(arg, *args, **kwds):
  return tf.get_collection(keys_dict[arg], *args, **kwds)

#-------------------------------------------------------------------------------
# Flags dictionary
flag_dict = {'auto_reuse': TF_AUTO_REUSE}

#-------------------------------------------------------------------------------
def Flag(arg):
  return flag_dict[arg]

#-------------------------------------------------------------------------------
# Summary dictionary
summary_dict = {'scalar': tf.summary.scalar, 'distribution': tf.summary.histogram}

#-------------------------------------------------------------------------------
def Summary(arg):
  return summary_dict[arg]

#-------------------------------------------------------------------------------
# Parameters list

Param_List = ['weights', 'biases']

#-------------------------------------------------------------------------------
# Logits list

Logits_List = [tf.nn.sparse_softmax_cross_entropy_with_logits, tf.nn.sigmoid_cross_entropy_with_logits]

#flags = {'reuse', tf.REUSE}
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Shape function as list

def Shape(X):
  XS = list(X.shape)
  S = [None] * len(XS)
  for i, xs in enumerate(XS):
    try:
      s = int(xs)
      S[i] = s
    except TypeError:
      pass
  return S

#-------------------------------------------------------------------------------

