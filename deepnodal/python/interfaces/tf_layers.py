# Tensorflow core layers are flexible, but not flexible enough

# Gary Bhumbra

#------------------------------------------------------------------------------- 
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape

#------------------------------------------------------------------------------- 
class tf_Dense(tf.layers.Dense):
  """
  Identical to tf.layer.Dense but initializers may be replaced with graph objects
  to share weight and/or bias parameters.

  """
  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None, 
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    tf.layers.Dense.__init__(self, units, activation, use_bias, 
                             kernel_initializer, bias_initializer,
                             kernel_regularizer, bias_regularizer, activity_regularizer, 
                             kernel_constraint, bias_constraint, 
                             trainable=trainable, name=name)

#------------------------------------------------------------------------------- 
  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    if type(self.kernel_initializer) is tf.Tensor:
      self.kernel = self.kernel_initializer
    else:
      self.kernel = self.add_variable('kernel',
                                      shape=[input_shape[-1].value, self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
    if self.use_bias:
      if type(self.bias_initializer) is tf.Tensor:
        self.bias = self.bias_initializer
      else:
        self.bias = self.add_variable('bias',
                                      shape=[self.units,],
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
    else:
      self.bias = None
    self.built = True
#------------------------------------------------------------------------------- 

#------------------------------------------------------------------------------- 
def tf_dense(
  inputs, units,
  activation=None,
  use_bias=True,
  kernel_initializer=None,
  bias_initializer=init_ops.zeros_initializer(),
  kernel_regularizer=None,
  bias_regularizer=None,
  activity_regularizer=None,
  kernel_constraint=None,
  bias_constraint=None,
  trainable=True,
  name=None,
  reuse=None):
  layer = tf_Dense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)

#------------------------------------------------------------------------------- 
def tf_dense2map(
  inputs, units,
  activation=None,
  use_bias=True,
  kernel_initializer=None,
  bias_initializer=init_ops.zeros_initializer(),
  kernel_regularizer=None,
  bias_regularizer=None,
  activity_regularizer=None,
  kernel_constraint=None,
  bias_constraint=None,
  trainable=True,
  name=None,
  reuse=None):
  if type(units) is not set:
    raise TypeError("Layer dense2hl units specification must of set type")
  units = list(units)
  if len(units) != 1:
    raise ValueError("Layer dense2hl units set specification must have one element")
  units = units[0]
  
  layer = tf_Dense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)

#------------------------------------------------------------------------------- 
class tf_Map2dense(base.Layer):
  """Sparse-to-dense lookup layer class

  This layer implements the operation
  `outputs = vergence(lookup(inputs) in kernel)`
  where `vergence` is the vergence function passed as the vergence argument
  (if not `None`), `kernel` is weights matrix created by the layer.

  Arguments
    cardinality: vocabulary set size
    units: integer or a pair of integers
    kernel_initializer: Initializer function for the kernel matrix.
      If `None` (default), weights are initialised using the default initializer
      used by `tf.get_variable`.
    trainable: Boolean, if `True' also add variables to teh graph collection
      `GraphKeys.TRAINABLE_VARIABLES`
    name: String

  """
#------------------------------------------------------------------------------- 
  def __init__(self, cardinality, units,
               kernel_initializer=None,
               kernel_dtype = tf.float32,
               trainable=True,
               name=None,
               **kwds):
    super(tf_Map2dense, self).__init__(trainable=trainable, name=name, **kwds)
    self.cardinality = cardinality
    self.units = units
    self.kernel_initializer = kernel_initializer
    self.kernel_dtype = kernel_dtype
    self.lookup_kwds = dict(kwds)
    self.input_spec = base.InputSpec(min_ndim=2)

#------------------------------------------------------------------------------- 
  def build(self, input_shape = None):
    input_shape = tensor_shape.TensorShape(input_shape)

    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Lookup` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2, axes={-1:input_shape[-1].value})

    kernel_width = self.units if type(self.units) is int else self.units[-1]
    self.kernel = self.add_variable('kernel',
                                    shape=[self.cardinality, kernel_width],
                                    dtype=self.kernel_dtype,
                                    initializer=self.kernel_initializer,
                                    trainable=True)
    self.built = True

#------------------------------------------------------------------------------- 
  def call(self, inputs):
    if len(inputs.shape) != 2:
      raise ValueError("Lookup input must be two-dimensional")
    unit_lookup = True if type(self.units) is int else len(self.units) == 1
    if unit_lookup:
      self.outputs = tf.nn.embedding_lookup(self.kernel, inputs, **self.lookup_kwds)
      self._output = self.outputs
      return self._output
    n_lookups = self.units[0]
    self.outputs = [None] * n_lookups
    for i in range(n_lookups):
      self.outputs[i] = tf.nn.embedding_lookup(self.kernel, inputs[:, i],
                                               **self.lookup_kwds)
    if type(self.units) is tuple:
      self._outputs = [None] * n_lookups
      for i in range(n_lookups):
        self._outputs[i] = tf.expand_dims(self.outputs[i], axis = 0)
      self._output = tf.reduce_sum(tf.concat(self._outputs, axis = 0), axis = 0)
      return self._output
    
    self._output = tf.concat(self.outputs, axis = -1)
    return self._output

#------------------------------------------------------------------------------- 
  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    kernel_width = units if type(self.units) is int else units[-1]
    unit_lookup = True if type(self.units) is int else len(self.units) == 1
    if unit_lookup or type(self.units) is tuple:
      return input_shape[:-1].concatenate(self.kernel_width)
    return input_shape[:-1].concatenate(self.units)

#------------------------------------------------------------------------------- 
#------------------------------------------------------------------------------- 
def tf_map2dense(inputs, cardinality, units, *args, **kwds):
  """ 
  tf_lookup - functional form of tf_Lookup
  inputs: inputs (integer data type)
  cardinality: vocabulary set size
  units: integer or a pair of integers
  kernel_initializer: Initializer function for the kernel matrix.
    If `None` (default), weights are initialised using the default initializer
    used by `tf.get_variable`.
  trainable: Boolean, if `True' also add variables to teh graph collection
    `GraphKeys.TRAINABLE_VARIABLES`
  name: String
  """
  layer = tf_Map2dense(cardinality, units, *args, **kwds)
  return layer.apply(inputs)

#------------------------------------------------------------------------------- 

