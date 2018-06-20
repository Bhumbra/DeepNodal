# Deepnodal interface for TensorFlow
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import name_scope, variable_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops.init_ops import VarianceScaling
from deepnodal.python.interfaces.tf_extended import *

#-------------------------------------------------------------------------------
# It seems tf.AUTO_REUSE does not feature in early TensorFlow versions
try:
  TF_AUTO_REUSE = tf.AUTO_REUSE
except AttributeError:
  TF_AUTO_REUSE = False

#-------------------------------------------------------------------------------
# Creation dictionary

creation_dict = {'identity': tf.identity,
                 'transpose': tf.transpose,
                 'ret_var': tf.get_variable,
                 'add': tf.add,
                 'add_ewise': tf.add_n,
                 'aug_dims': tf.expand_dims,
                 'assign': tf.assign,
                 'combine': tf.group,
                 'deps': tf.control_dependencies,
                 'subtract': tf.subtract,
                 'multiply': tf.multiply,
                 'shape': tf.shape,
                 'mean': tf.reduce_mean,
                 'sum': tf.reduce_sum,
                 'con': tf.concat,
                 'pack': tf.stack,
                 'matmul': tf.matmul,
                 'eye': tf.eye,
                 'cast': tf.cast,
                 'diverge': tf.split,
                 'verge': tf_vergence,
                 'dense': tf_dense,
                 'conv1d': {'xcorr': tf.layers.conv1d,
                            'sconv': tf.layers.separable_conv1d,
                            'tconv': None}, # Force an error
                 'conv2d': {'xcorr': tf.layers.conv2d,
                            'sconv': tf.layers.separable_conv2d,
                            'tconv': tf.layers.conv2d_transpose},
                 'conv3d': {'xcorr': tf.layers.conv3d,
                            'sconv': None, # Force an error
                            'tconv': tf.layers.conv3d_transpose},
                 'pool1d': {'max':tf.layers.max_pooling1d,
                            'avg':tf.layers.average_pooling1d},
                 'pool2d': {'max':tf.layers.max_pooling2d,
                            'avg':tf.layers.average_pooling2d},
                 'pool3d': {'max':tf.layers.max_pooling3d,
                            'avg':tf.layers.average_pooling3d},
                 'flatten': tf.layers.flatten,
                 'lookdown': tf_lookdown,
                 'lookup': tf_lookup,
                 #'batch_norm': tf.layers.batch_normalization,
                 'batch_norm': batch_norm,
                 'lresp_norm': tf.nn.local_response_normalization,
                 'dropout': tf.layers.dropout,
                 'l1_reg': tf_l1_loss,
                 'l2_reg': tf_l2_loss,
                 'max_norm': tf_max_norm_regularizer,
                 'relu': tf.nn.relu,
                 'elu': tf.nn.elu,
                 'softmax': tf.nn.softmax,
                 'sigmoid': tf.nn.sigmoid,
                 'var': tf.Variable,
                 'tensor': tf.placeholder,
                 's2d': tf.sparse_to_dense,
                 'vs': VarianceScaling, # TensorFlow's version
                 'vsi': tf_variance_scaling_initialiser, # mine
                 'zoi': tf.zeros_initializer,
                 'lvi': tf.local_variables_initializer,
                 'gvi': tf.global_variables_initializer,
                 'sgd': tf.train.GradientDescentOptimizer,
                 'mom': tf.train.MomentumOptimizer,
                 'adagrad': tf.train.AdagradOptimizer,
                 'rmsprop': tf.train.RMSPropOptimizer,
                 'adam': tf.train.AdamOptimizer,
                 'in_top_k_error': tf_in_top_k_error,
                 'mse': tf.losses.mean_squared_error,
                 'mce': tf_mean_cross_entropy,
                 'nce': tf.nn.nce_loss,
                 'onehot': tf.one_hot,
                 'logger': tf.summary.FileWriter,
                 'saver': tf.train.Saver,
                 'defaults': tf.get_default_graph,
                 'session': tf.Session}

def Creation(*args):
  if not(len(args)): return None
  creation = creation_dict
  for arg in args:
    creation = arg if type(arg) is not str else creation[arg.lower()]
  return creation

#-------------------------------------------------------------------------------
# Dtype 

dtype_dict = {None: None,
              'bool': tf.bool,
              'int32': tf.int32,
              'int64': tf.int64,
              'float32': tf.float32,
              'float64': tf.float64,
              'tensor': tf.Tensor}

def Dtype(arg):
  return dtype_dict[arg]


#-------------------------------------------------------------------------------
# Scope 

scope_dict = {'name': name_scope,
              'var': variable_scope,
              'arg': arg_scope}

def Scope(spec, *args, **kwds):
  return scope_dict[spec](*args, **kwds)

#-------------------------------------------------------------------------------
# Keys (not yet in use in deepnodal)
keys_dict = {'reg': tf.GraphKeys.REGULARIZATION_LOSSES}

def Keys(arg, *args, **kwds):
  return tf.get_collection(keys_dict[arg], *args, **kwds)

#-------------------------------------------------------------------------------
# Flags 
flag_dict = {'auto_reuse': TF_AUTO_REUSE}

def Flag(arg):
  return flag_dict[arg]

#-------------------------------------------------------------------------------
# Summary 
summary_dict = {'scalar': tf.summary.scalar, 'distro': tf.summary.histogram}

def Summary(arg):
  return summary_dict[arg]

#-------------------------------------------------------------------------------
# Device
device_dict = {'device': tf.device, 
               'cpu': '/device:CPU:', 
               'gpu': '/device:GPU:'}

def Device(spec = None, number = None): # returns a string if number is not None
  if number is None: return device_dict['device'](spec)
  if spec == 'cpu' and number != 0:
    raise ValueError("TensorFlow support for only single CPU device")
  return device_dict[spec] + str(number)

#-------------------------------------------------------------------------------
# Parameters list

Param_Dict = {'kernel': 'weights',  'bias': 'biases'}
Param_Reg = {'weights': 'kernel'}

#-------------------------------------------------------------------------------
# Logits list

Logits_List = [tf.nn.softmax, tf.nn.sigmoid]

#flags = {'reuse', tf.REUSE}
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Shape function as list

def Shape(X):
  S = [None]
  XS = X.shape
  try:
   XS = list(XS)
  except ValueError:
    return S
  S = [None] * len(XS)
  for i, xs in enumerate(XS):
    try:
      s = int(xs)
      S[i] = s
    except TypeError:
      pass
  return S

#-------------------------------------------------------------------------------

