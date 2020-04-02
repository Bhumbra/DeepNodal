# Deepnodal interface for TensorFlow
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import name_scope
from tensorflow.compat.v1 import variable_scope
from deepnodal.python.interfaces.arg_scope import arg_scope
from tensorflow.python.ops.init_ops import VarianceScaling
from deepnodal.python.interfaces.tf_extended import *

#-------------------------------------------------------------------------------
# It seems tf.AUTO_REUSE does not feature in early TensorFlow versions
try:
  TF_AUTO_REUSE = tf.compat.v1.AUTO_REUSE
except AttributeError:
  TF_AUTO_REUSE = False

#-------------------------------------------------------------------------------
# Creator dictionary
creator_dict = {
                 'batch_norm': tf.keras.layers.BatchNormalization,
                 'recurrent': tf.keras.layers.RNN,
                 'flatten': tf.keras.layers.Flatten,
                 'dropout': tf.keras.layers.Dropout,
               }

#-------------------------------------------------------------------------------
def Creator(*args):
  if not(len(args)): return None
  creator = creator_dict
  for arg in args:
    creator = arg if type(arg) is not str else creator[arg.lower()]
  return creator

#-------------------------------------------------------------------------------
# Creation dictionary

creation_dict = {'identity': tf.identity,
                 'transpose': tf.transpose,
                 'expand_dims': tf.expand_dims,
                 'ret_var': tf.compat.v1.get_variable,
                 'add': tf.add,
                 'add_ewise': tf.add_n,
                 'aug_dims': tf.expand_dims,
                 'assign': tf.compat.v1.assign,
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
                 'recurrent': tf.keras.layers.RNN,
                 'squeeze': tf.squeeze,
                 'dense2card': tf_dense2card,
                 'card2dense': tf_card2dense,
                 'dropout': tf.layers.dropout,
                 'batch_norm': tf.layers.batch_normalization,
                 'rec': tf.keras.layers.SimpleRNNCell,
                 'gru': tf.keras.layers.GRUCell,
                 'lstm': tf.keras.layers.LSTMCell,
                 'lresp_norm': tf.nn.local_response_normalization,
                 'l1_reg': tf_l1_loss,
                 'l2_reg': tf_l2_loss,
                 'weight_decay': tf_weight_decay,
                 'max_norm': tf_max_norm,
                 'relu': tf.nn.relu,
                 'elu': tf.nn.elu,
                 'softmax': tf.nn.softmax,
                 'sigmoid': tf.nn.sigmoid,
                 'var': tf.Variable,
                 'tensor': tf.compat.v1.placeholder,
                 's2d': tf.sparse_to_dense,
                 'vs': VarianceScaling, # TensorFlow's version
                 'vsi': tf_variance_scaling_initialiser, # mine
                 'zoi': tf.zeros_initializer,
                 'lvi': tf.compat.v1.local_variables_initializer,
                 'gvi': tf.compat.v1.global_variables_initializer,
                 'sgd': tf.compat.v1.train.GradientDescentOptimizer,
                 'mom': tf.compat.v1.train.MomentumOptimizer,
                 'adagrad': tf.compat.v1.train.AdagradOptimizer,
                 'rmsprop': tf.compat.v1.train.RMSPropOptimizer,
                 'adam': tf.compat.v1.train.AdamOptimizer,
                 'in_top_k_error': tf_in_top_k_error,
                 'mse': tf.compat.v1.losses.mean_squared_error,
                 'mce': tf_mean_cross_entropy,
                 'nce': tf.nn.nce_loss,
                 'onehot': tf.one_hot,
                 'logger': tf.compat.v1.summary.FileWriter,
                 'saver': tf.compat.v1.train.Saver,
                 'defaults': tf.compat.v1.get_default_graph,
                 'session': tf.compat.v1.Session}

#-------------------------------------------------------------------------------
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
# Return variable withins cope if exists
def Ret_Var(scope, name):
  var = None
  if isinstance(scope, str):
    with Scope('var', scope, reuse=True):
      try:
        var = Creation('ret_var')(name)
      except ValueError:
        return var
    return var
  assert isinstance(scope, (list, tuple)), "Scope must be str, list or tuple"
  for variable in scope:
    if name in variable.name:
      assert var is None, "Mutiple variables found in scope for {}".format(name)
      var = variable
  return var

#-------------------------------------------------------------------------------
# Keys (not yet in use in deepnodal)
keys_dict = {'reg': tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES}

def Keys(arg, *args, **kwds):
  return tf.compat.v1.get_collection(keys_dict[arg], *args, **kwds)

#-------------------------------------------------------------------------------
# Flags 
flag_dict = {'auto_reuse': TF_AUTO_REUSE}

def Flag(arg):
  return flag_dict[arg]

#-------------------------------------------------------------------------------
# Summary 
summary_dict = {'scalar': tf.compat.v1.summary.scalar, 'distro': tf.compat.v1.summary.histogram}

def Summary(arg):
  return summary_dict[arg]

#-------------------------------------------------------------------------------
# Device
device_dict = {'device': tf.device, 
               'cpu': '/device:CPU:', 
               'gpu': '/device:GPU:'}

def Device(spec = None, number = None): # returns a string if number is not None
  if number is None: return device_dict['device'](spec)
  return device_dict[spec] + str(number)

#-------------------------------------------------------------------------------
# Parameters list

Param_Dict = {'kernel': 'weights',  'bias': 'biases'}
Norm_Moments = {'moving_mean', 'moving_variance', 'renorm_mean', 'renorm_stddev'}
Norm_Dict = {'gamma': 'scale', 'beta': 'offset', \
             'renorm_mean_weight': 'renorm_offset', 'renorm_stddev_weight': 'renorm_scale'}
Param_Reg = {'weights': 'kernel'}
Regularisation = {'loss': [Creation('l1_reg'), Creation('l2_reg')],
                  'grad': [Creation('weight_decay')],
                  'vars': [Creation('max_norm')]}

#-------------------------------------------------------------------------------
def Updates(scope=None):
  if isinstance(scope, str):
    return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope=scope)
  return scope

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
# Random seeed
def Seed(n):
  return tf.compat.v1.set_random_seed(n)

#-------------------------------------------------------------------------------
