# Stream module for tensorflow

# Gary Bhumbra

#-------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
from deepnodal.tf_extended import *
from deepnodal.concepts import *
from tf_extended import *
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d, avg_pool2d
average_pooling2d = tf.layers.average_pooling2d
from tensorflow.contrib.layers import flatten, batch_norm, l1_regularizer, l2_regularizer, dropout
local_response_normalization = tf.nn.local_response_normalization
from tensorflow.contrib.framework import arg_scope
from tensorflow import name_scope, variable_scope

#-------------------------------------------------------------------------------
class stream (structure):
  # a stream is a sequential train with a single transform operation
  # that may involve weights and biases parameters
  inp = None # input
  reg = None # regulariser (L1 or L2 or max-norm)
  out = None # output (final result - not data)
  trf = None # transfer function
  params_ari = None # arithmetic parameters (i.e. biases)
  params_geo = None # geometric parameters (i.e. weights)
  params_exp = None # exponential parameters (i.e. powers) - not in use currently
  pretransform = None # supports dropout
  transformer = None # supports dense, conv, and pool
  postransform = None # supports LRN
  params_ops = None   # operations to set parameters
  
#-------------------------------------------------------------------------------
  def __init__(self, name = 'stream', dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.initialise()

#-------------------------------------------------------------------------------
  def initialise(self):
    self.set_inp()
    self.set_var()
    self.set_reg()
    self.add_pretransform()
    self.set_transformer()
    self.add_postransform()

#-------------------------------------------------------------------------------
  def set_name(self, name = 'stream'):
    self.name = name

#-------------------------------------------------------------------------------
  def set_reg(self, reg = None, *_reg_args, **_reg_kwargs):
    self.reg = reg
    if self.reg is None:
      pass
    elif type(self.reg) is bool:
      self.reg = max_norm_regularizer if self(reg) else None
    elif self.reg == 1: 
      self.reg = l1_regularizer
    elif self.reg == 2: 
      self.reg = l2_regularizer
    else:
      pass
    self.reg_args = _reg_args
    self.reg_kwargs = _reg_kwargs 

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def setup_input(self, inp, inp_flatten = False):
    self.inp = inp  # stream claims no ownership over input,
    if type(self.inp) is not tf.Tensor:
      raise TypeError("Input type must be a tensor.")
    if inp_flatten: # but will claim ownership over flattening operation
      if self.dev is None:
        self.inp = flatten(inp, scope=self.name+"/input_flatten")
      else:
        with tf.device(self.dev):
          self.inp = flatten(inp, scope=self.name+"/input_flatten")
    return self.retout()

#-------------------------------------------------------------------------------
  def add_pretransform(self, func_call = None, *args, **kwargs):
    if func_call is None:
      if self.pretransform is None:
        self.pretransform = []
      return self.retout()

    Inp = self.inp 
    if self.pretransform is not None:
      if len(self.pretransform):
        Inp = self.pretransform[-1]
    if self.dev is None:
      self.pretransform.append(func_call(Inp, *args, **kwargs))
    else:
      with tf.device(self.dev):
        self.pretransform.append(func_call(Inp, *args, **kwargs))
    return self.retout()

#-------------------------------------------------------------------------------
  def set_transformer(self, func_call = None, *args, **kwargs):
    if func_call is None:
      self.transformer = None
      return self.retout()

    Inp = self.inp 
    if self.pretransform is not None:
      if len(self.pretransform):
        Inp = self.pretransform[-1]
    kwds = dict(kwargs)
    self.trf = None if 'activation_fn' not in kwds else kwds['activation_fn']
    if self.dev is None:
      if self.reg is None:
        self.transformer = func_call(Inp, *args, scope = self.name, **kwds)
      else:
        with arg_scope ([func_call], weights_regularizer = self.reg(*self.reg_args, **self.reg_kwargs)):
          self.transformer = func_call(Inp, *args, scope = self.name, **kwds)
    else:
      with tf.device(self.dev):
        if self.reg is None:
          self.transformer = func_call(Inp, *args, scope = self.name, **kwds)
        else:
          with arg_scope ([func_call], weights_regularizer = self.reg(*self.reg_args, **self.reg_kwargs)):
            self.transformer = func_call(Inp, *args, scope = self.name, **kwds)
    if self.transformer is not None:
      with variable_scope(self.name, reuse=True):
        try:
          self.params_ari = tf.get_variable('biases', initializer=tf.zeros_initializer())
          self.params_geo = tf.get_variable('weights')
        except ValueError:
          pass
    return self.retout() # this will update self.out and output self.transfomer

#-------------------------------------------------------------------------------
  def add_postransform(self, func_call = None, *args, **kwargs):
    if func_call is None:
      if self.postransform is None:
        self.postransform = []
      return self.retout()

    Inp = self.retout()
    if self.dev is None:
      if func_call is tf.concat:
        self.postransform.append(func_call([Inp]+list(args), **kwargs))
      else:
        self.postransform.append(func_call(Inp, *args, **kwargs))
    else:
      with tf.device(self.dev):
        if func_call is tf.concat:
          self.postransform.append(func_call([Inp]+list(args), **kwargs))
        else:
          self.postransform.append(func_call(Inp, *args, **kwargs))
    return self.retout()

#-------------------------------------------------------------------------------
  def ret_out(self):
    self.out = self.inp
    if self.pretransform is not None: 
      if len(self.pretransform):
        self.out = self.pretransform[-1]
    if self.transformer is not None: 
      self.out = self.transformer
    if self.postransform is not None: 
      if len(self.postransform):
        self.out = self.postransform[-1]
    return self.out

#-------------------------------------------------------------------------------
  def vsi_weights(self, session, trf_map = None, distr_map = {}, locan_map = {}, scale_map = {}, shape_map = {}): 
    # variance-scale-initialise, parameters can be mapped to transfer function
    if self.params_geo is None: return
    Ws = np.array(self.params_geo.eval().shape, dtype = int)

    # Default to He et al. initialisation
    fan = 'fan_in'
    distr = 'trun'
    locan = 0.
    scale = 1.
    shape = 2.
    map_trf = trf_map if type(trf_map) is bool else trf_map
    if map_trf is None: map_trf = False
    if map_trf:
      if type(map_trf) is bool:
        map_trf = self.trf
      try:
        distr = distr_map[map_trf]
        locan = locan_map[map_trf]
        scale = scale_map[map_trf]
        shape = shape_map[map_trf]
      except KeyError: # default to He et al. initialisation
        pass
    W = tf_weight_initialiser(Ws, fan=fan, distr=distr, locan=locan, scale=scale, shape=shape)
    return self.set_vars(session, weights = W)

#-------------------------------------------------------------------------------
  def set_param_ops(self, params_ari = None, params_geo = None, remember = True):
    # params_exp is not yet supported
    param_ops = [tf.no_op(name=self.name + "/no_op"), tf.no_op(name=self.name + "/no_op")]
    if self.param_ari is not None and params_ari is not None:
      param_ops[0] = tf.assign(self.params_ari, params_ari, name = self.name + "/biases_assign")
    if self.param_geo is not None and params_geo is not None:
      param_ops[1] = tf.assign(self.params_geo, params_geo, name = self.name + "/weights_assign")
    param_ops = tf.group(set[0], setvarsOp[1], name=self.name + "/params_assign")
    if not remember: return param_ops
    self.param_ops = param_ops
    return self.param_ops

#-------------------------------------------------------------------------------
  def set_vars(self, session = None, weights = None, biases = None):
    if session is None:
      raise ValueError("Cannot set parameters without session in progress")
    ops = self.setvarsop(weights, biases, False)
    session.run(ops)

#-------------------------------------------------------------------------------
  def ret_vars(self):
    _weights = None if self.params_geo is None else self.params_geo.eval()
    _biases = None if self.params_ari is None else self.params_ari.eval()
    return _weights, _biases

#-------------------------------------------------------------------------------

