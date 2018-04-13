# Functions to extend TensorFlow with a homogenised interface

# Gary Bhumbra

import tensorflow as tf
import numpy as np
from scipy import stats
from tensorflow.contrib.layers import max_pool2d, avg_pool2d
average_pooling2d = tf.layers.average_pooling2d
from tensorflow import name_scope, variable_scope

#-------------------------------------------------------------------------------
# TensorFlow has no max_norm regulariser

def tf_max_norm_regularizer(clip_norm, axes = 1., name = "max_norm", collection = "max_norm"):
  def max_norm(weights):
    clipped = tf.clipped(weights, clip_norm = clip_norm, axes = axes)
    clip_weights = tf.assign(weights, clipped, name = name)
    tf.add_to_collection(collection, clip_weights)
    return None
  return maxnorm

#-------------------------------------------------------------------------------
# TensorFlow has a different interfaces for different pooling layers 

def tf_pool2d(*args, **kwds):
  tf_func = {'max':max_pool2d, 'avg':avg_pool2d, 'gap':average_pooling2d}
  try:
    pool_type = kwds['pool_type']
    del kwds['pool_type']
  except KeyError:
    pool_type = 'max'
  pool_func = tf_func[pool_type]
  kwargs = dict(kwds)
  try:
    pool_act = kwds['activation_fn']
    del kwargs['activation_fn']
  except KeyError:
    pool_act = None
  if pool_type == 'gap':
    kwargs['pool_size'] = kwargs.pop('kernel_size')
    kwargs['strides'] = kwargs.pop('stride')
    if 'scope' in kwargs:
      kwargs['name'] = kwargs.pop('scope')
  pool_out = pool_func(*args, **kwargs)
  if pool_act is None: return pool_out
  try:
    pool_name = kwds['scope']
  except KeyError:
    return pool_act(pool_out)
  with name_scope(pool_name+"/"):
    pool_actn = pool_act(pool_out)
  return pool_actn

#-------------------------------------------------------------------------------
# TensorFlow has different interfaces for joining inputs either as sum of concat

def tf_coalesce(X, coalescence_fn = 'con', axis = -1, **kwargs): 
  # valid values are 'con' for concatenate and 'sum' for add
  if coalescence_fn == 'con':
    return tf.concat(X, axis = axis, **kwargs)
  if coalescence_dn == 'sum':
    return tf.reduce_sum(X, axis = axis, **kwargs)
  raise ValueError("Unknown coalescence function specification: " + str(coalescence_dn))

#-------------------------------------------------------------------------------
# TensorFlow's weight initialisation supports very little customisation

def tf_weight_initialiser(_W_shape, fan='in', distr = 'unif', locan=0., scale = 1., shape=1., dtype = np.float):
  W_shape = np.atleast_1d(_W_shape)
  fan = fan.lower()
  distr = distr.lower()
  locan = float(locan)
  scale = float(scale)
  shape = float(shape)

  n_input = W_shape[-2]
  n_output = W_shape[-1]
  if len(W_shape) > 2:
    prod_dim = np.prod(W_shape[:-2])
    n_input *= prod_dim
    n_output *= prod_dim
  if fan == 'out':
    n = n_output
  elif fan == 'in':
    n = n_input
  else:
    n = 0.5 * (n_input + n_output)

  W = None
  mn = locan / n
  sd = np.sqrt(2./n) # a la he2015delving

  if distr == 'norm':
    W = np.random.normal(0., scale=sd, size=W_shape)
  elif distr == 'unif': # this corresponds to an sd of np.sqrt(1./n) not np.sqrt(2./n)
    limit = np.sqrt(3. / n)
    W = np.random.uniform(-limit, limit, size=W_shape)
  elif distr == 'trun':
    W = sd * stats.truncnorm.rvs(-shape, shape, size=W_shape) * 1.137
  elif distr == 'gamm':
    lmbda = sd / np.sqrt(shape)
    W =  np.random.gamma(shape, lmbda, size=W_shape)
    W -= lmbda * shape
  elif distr == 'gamd':
    lmbda = sd / np.sqrt(2. * shape)
    W =  np.random.gamma(shape, lmbda, size=W_shape)
    W -= np.random.gamma(shape, lmbda, size=W_shape)
  elif distr == 'beta':
    W = sd * (np.random.beta(shape, shape, size=W_shape)-0.5) / np.sqrt(1. / (8.*shape + 4.))
  elif distr == 'bgam':
    lmbda = sd / np.sqrt(shape * (shape + 1.))
    W_shape_prod = int(np.prod(W_shape))
    W_number_plus = W_shape_prod // 2
    W_number_minus = W_shape_prod - W_number_plus
    W = np.hstack( [np.random.gamma(shape, lmbda, size=W_number_plus),
                   -np.random.gamma(shape, lmbda, size=W_number_minus)]  )
    W = W[np.random.permutation(W_shape_prod)].reshape(W_shape)
  elif distr == 'tria':
    W = sd * (np.random.triangular(0., 0.5, 1., size=W_shape) - 0.5) * np.sqrt(24.)

  if W is None: raise ValueError("Unknown distribution specification: " + dist)
  return scale * np.array(W, dtype = dtype) + locan / n

#-------------------------------------------------------------------------------
