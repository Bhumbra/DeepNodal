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
def tf_l1_loss(t, name=None):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.reduce_sum(tf.abs(t))

#-------------------------------------------------------------------------------
def tf_in_top_k_error(X, labels, k = 1, dtype = tf.float32, name = None):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.subtract(1., tf.reduce_mean(tf.cast(tf.nn.in_top_k(X, labels, k), dtype)))

#-------------------------------------------------------------------------------
def tf_mean_cross_entropy(logits, labels, activation_fn, name = None):
  func_dict = {tf.nn.sigmoid: tf.nn.sigmoid_cross_entropy_with_logits,
               tf.nn.softmax: tf.nn.sparse_softmax_cross_entropy_with_logits}
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.reduce_mean(func_dict[activation_fn](logits=logits, labels=labels))

#-------------------------------------------------------------------------------
def tf_vergence(X, vergence_fn = 'con', divergence = None, axis = -1, **kwargs): 
  # valid values are 'con' for concatenate and 'sum' for add
  if vergence_fn == 'con':
    return tf.concat(X, axis = axis, **kwargs)
  if vergence_fn == 'sum':
    return tf.reduce_sum(X, axis = axis, **kwargs)
  if vergence_fn == 'div':
    if divergence is None:
      raise ValueError("Vergence with vergence_fn='div' requires divergence specification")
    return tf.split(X, divergence, axis = axis, **kwargs)
  raise ValueError("Unknown vergence function specification: " + str(coalescence_dn))

#-------------------------------------------------------------------------------
# TensorFlow contrib's weight initialisation supports very little customisation

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
