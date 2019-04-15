# Functions to extend TensorFlow with a homogenised interface

# Gary Bhumbra

import numpy as np
import math
from deepnodal.python.interfaces.tf_layers import *
from tensorflow import name_scope, variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.init_ops import Initializer, _compute_fans

#-------------------------------------------------------------------------------
def tf_l1_loss(t, name=None, scale = 1.):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.multiply(tf.reduce_sum(tf.abs(t)), scale)

#-------------------------------------------------------------------------------
def tf_l2_loss(t, name=None, scale = 1.):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    #return tf.multiply(tf.nn.l2_loss(t), scale) # Cannot replicate
    return tf.multiply(tf.divide(tf.reduce_sum(tf.square(t)), 2), scale)

#-------------------------------------------------------------------------------
def tf_weight_decay(t, scale=1., name='weight_decay'):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.multiply(scale, t)

#-------------------------------------------------------------------------------
def tf_max_norm(weights, clip_norm, axes=1., name='max_norm'):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    return tf.clipped(weights, clip_norm=clip_norm, axes=axes)

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
def tf_cosine_similarity(data, weights, name=None):
  with variable_scope(name, reuse=tf.AUTO_REUSE):
    normalised_weights = weights / tf.sqrt(tf.reduce_sum(tf.square(weights), 1, keepdims=True))
    data_map2dense = tf.nn.embedding_lookup(normalised_weights, data)
    return tf.matmul(data_map2dense, normalised_weights, tranpose_b=True)

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
# TensorFlow weight initialisation supports very little customisation

# from tensorflow.python.ops.init_ops import Initializer
class tf_variance_scaling_initialiser (Initializer):
  """
  An initialiser that generates tensor for weight parameters.
  distr is the chosen distribution which may be 'unif' (uniform), 'norm', 'trun' (truncated normal), 'gamma' or 'beta'
  fan specifies the normalisation denominator and can be 'in'. 'out', or 'avg'.
  locan is the offset numerator to centralise the distribution (default 0, but for example you could use -1).
  scale is the scaling factor prior to offsetting (default 1.).
  shape (ignored if distr = 'unif' or 'norm' or 'trun') is the shaping parameter for 'gamma' and 'beta')

  This code tries to match the functionality of tensorflow.python.ops.init_ops.VarianceScaling.
  """
  def __init__(self, distr = 'trun', fan = 'in', locan = 0., scale = 1., shape = 1.,
      seed = None, dtype = dtypes.float32):
    self.set_config(distr, fan, locan, scale, shape, seed, dtype)

#-------------------------------------------------------------------------------
  def set_config(self, distr = 'trun', fan = 'in', locan = 0., scale = 1., shape = 1.,
      seed = None, dtype = dtypes.float32):
    self.distr, self.fan = distr.lower(), fan.lower()
    if scale <= 0.:
      raise ValueError("The scale argument must be a positive floating point")
    if self.distr not in ['unif', 'norm', 'trun', 'gamma', 'beta']:
      raise ValueError("Invalid fan specification: ", self.distr)
    if self.fan not in ['in', 'out', 'avg']:
      raise ValueError("Invalid fan specification: ", self.fan)
    self.locan, self.scale, self.shape = locan, scale, shape
    self.seed, self.dtype = seed, dtype

#-------------------------------------------------------------------------------
  def __call__(self, shape, dtype=None, partition_info=None):
    # Note here, shape refers to dims not distribution shape parameter!

    # We will switch in one line:
    dims = shape if partition_info is None else partition_info.full_shape

    if dtype is None:
      dtype = self.dtype
    if not dtype.is_floating:
      raise TypeError("Only floating data types supported.")

    # Calculate denominator
    ninp, nout = _compute_fans(dims)
    n = None
    if self.fan == 'in':
      n = max(1., ninp)
    elif self.fan == 'out':
      n = max(1., nout)
    elif self.fan == 'avg':
      n = max(1., 0.5 * (ninp + nout))

    # Random number generator
    rand = None
    stdev = math.sqrt(2. / n) # a la he2015delving

    if self.distr == 'unif':
      limit = stdev * math.sqrt(3.)
      rand = random_ops.uniform(dims, -limit, limit, self.dtype, seed=self.seed)
    elif self.distr == 'norm':
      rand = random_ops.random_normal(dims, 0., stdev, self.dtype, seed=self.seed)
    elif self.distr == 'trun': # -2SD to +2SD truncated normal has a SD of 1./1.137
      rand = random_ops.truncated_normal(dims, 0., 1.137*stdev, self.dtype, seed=self.seed)
    elif self.distr == 'gamma':
      gamma = self.shape
      invlb = math.sqrt(gamma) / stdev
      rand = random_ops.random_gamma(dims, gamma, invlb, self.dtype, seed=self.seed) - (gamma/invlb)
    elif self.distr == 'beta':
      gamma = self.shape
      rand_0 = random_ops.random_gamma(dims, gamma, 1., self.dtype, seed=self.seed)
      rand_1 = random_ops.random_gamma(dims, gamma, 1., self.dtype, seed=self.seed)
      rand = stdev * ((rand_0 / (rand_0 + rand_1)) - 0.5) * math.sqrt(8.*self.shape + 4.)

    # Scale and offset as required
    randscaled = rand if self.scale == 1. else rand * self.scale
    randoffset = randscaled if self.locan == 0. else randscaled + (self.locan/n)

    return randoffset

#-------------------------------------------------------------------------------
  def ret_config(self):
    return {
        'distr': self.distr,
        'fan': self.fan,
        'locan': self.locan,
        'scale': self.scale,
        'shape': self.shape,
        'seed': self.seed,
        'dtype': self.dtype,
    }

#-------------------------------------------------------------------------------
  def get_config(self): # TensorFlow's convention
    return self.ret_config()

#-------------------------------------------------------------------------------
