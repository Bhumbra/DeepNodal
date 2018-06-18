# Tensorflow core layers are flexible, but not flexible enough

# Gary Bhumbra

#------------------------------------------------------------------------------- 
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape

#------------------------------------------------------------------------------- 
class tf_Dense(tf.layers.Dense):
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

