# Stream module for TensorFlow which abstracts out the unintuitive TensorFlow syntax
# to something more consistent.

# Gary Bhumbra

#-------------------------------------------------------------------------------
import numpy as np
from deepnodal.python.structures.chain import *
from deepnodal.python.concepts.structure import *

#-------------------------------------------------------------------------------
DEFAULT_STREAM_ORDER = 'datn' # over-ruled to only 'a' for absent architectures.
DEFAULT_BIASES = True
DEFAULT_TRANSFER_FUNCTION = None
DEFAULT_CONVOLUTION_KERNEL_FUNCTION = 'xcorr' # others: 'tconv' 'sconv'
DEFAULT_POOLING_KERNEL_FUNCTION = 'max'       # other: 'avg'
DEFAULT_PADDING_WINDOW = 'same'

#-------------------------------------------------------------------------------
class stream (chain):
  """
  a stream is an architectured chain supporting the following sequence in any order:

  d - dropout
  a - architectural operation that may involve training parameters (includes pooling)
  t - transfer function
  n - normalisation (i.e. lr_norm or batch_norm)

  there is strictly only one input and one output.

  There may be a maximum of 5 rather than 4 links in the chain since input
  redimensionalisation may precede the above list.

  self.set_`function' sets parameters involved with the stream design but does not
  create any TensorFlow objects.

  self.__call__`function' generates TensorFlow objects but should only be called via
  self.__call__(inp), which should be inkoved by a function-derived class.
  """

  # public
  def_name = 'stream' # default name
  arch = None         # architecture
  arch_args = None    # optional archetectural args for when arch is callable
  arch_kwds = None    # optional archetectural kwds for when arch is callable
  arch_link = None    # architecture link
  arch_out = None     # architecture output (i.e. raw weighted sum or pool result)
  type_arch = None    # architecture type without dimension
  type_adim = None    # architecture type with dimension
  order = None        # string denoting order of operation (default 'dotn')
  ist = None          # is_training flag
  bia = None          # Biases
  wgt = None          # Weights
  dro = None          # Dropout
  tfn = None          # Transfer function
  win = None          # Padding window
  kfn = None          # Kernel function (for convolution and pooling layers)
  nor = None          # Normalisation
  reg = None          # Regularisation
  vsi = None          # An instance of custom weights initialiser class if needed
  trans_fn = None     # trans_fn = tfn, used as a summary tf by higher objects
  dropout_quotient = None # Graph object for the dropout coefficient

  # protected
  _reguln = None      # list of dictionaries with parameter regularisation spec

#-------------------------------------------------------------------------------
  def __init__(self, name = 'stream', dev = None):
    chain.__init__(self, name, dev)
    self.set_arch() # defaults to 'none'

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None):
    """
    arch = None or []:  identity
    arch = integer:     dense
    arch = list of 1:   recurrent
    arch = list of 2:   pool
    arch = list of 3:   conv
    arch = dict  {sparse_length: [outer_dim, inner_dim]}: map2dense
    arch = set: dense2map

    """
    # Note links are not created here because they are added sequentially at
    # the stream.setup(inp) stage.

    self.arch = arch # arch = None is the default signifying a 'none'
    self.arch_link = None
    self.arch_out = None
    self.type_arch = None
    self.type_adim = None
    if self.arch is None:
      self.type_arch = 'none'
      self.type_adim = self.type_arch
    elif callable(self.arch):
      self.type_arch = 'callable'
      self.type_adim = self.arch
    elif type(self.arch) is int:
      self.type_arch = 'dense'
      self.type_adim = self.type_arch
    elif type(self.arch) is set:
      self.type_arch = 'dense'
      self.type_adim = 'dense2map'
      if len(self.arch) != 1:
        raise ValueError("Any dense2map archecture specification must be a single element set")
    elif type(self.arch) is dict:
      self.type_arch = 'map2dense'
      if len(self.arch) != 1:
        raise ValueError("Sparse to dense specification requires one dictionary element")
      arch_key = list(self.arch)[0]
      arch_val = self.arch[arch_key]
      arch_dim = 1
      if type(arch_val) is list:
        arch_dim = len(arch_val)
        if arch_dim != 1 and arch_dim != 2:
          raise ValueError("Dense specification must be one or two dimensional")
      self.type_adim = self.type_arch + str(arch_dim) + 'd'
    elif type(self.arch) is not list:
      raise ValueError("Unknown architecture specification")
    else:
      if len(arch) == 0:
        self.type_arch = 'identity'
        self.type_adim = self.type_arch
      elif len(self.arch) == 1:
        raise ValueError("List length of 1 reserved (for recurrent architectures)")
      elif len(self.arch) > 3:
        raise ValueError("Unknown architecture specification")
      else:
        if type(self.arch[1]) is not list:
          raise ValueError("Unknown architecture specification")
        self.type_arch = 'pool' if type(self.arch[0]) is list else 'conv'
        self.type_adim = self.type_arch + str(len(self.arch[1])) + "d"
        if self.type_arch == 'conv': # default to unit stride
          if len(self.arch) == 2:
            self.arch = [self.arch[0], self.arch[1], 1]
          if type(self.arch[2]) is int:
            self.arch[2] = [self.arch[2]] * len(self.arch[1])
        if self.type_arch == 'pool':
          if type(self.arch[1]) is int:
            self.arch[1] = [self.arch[1]] * len(self.arch[0])
    if self._parent is not None:
      self._parent.update_arch()

    # Default bare essentials
    self.set_order()
    self.set_arch_args()
    self.set_biases()
    self.set_transfn()
    self.set_padwin()
    self.set_kernfn()
    return self.type_arch

#-------------------------------------------------------------------------------
  def add_arch(self, *args, **kwds):
    raise AttributeError("Method stream.add_arch() not supported by stream class.")

#-------------------------------------------------------------------------------
  def set_is_training(self, ist = None):
    """
    ist = is_training must be set to handle some operations (e.g. batch normalisation)
    """
    self.ist = ist

#-------------------------------------------------------------------------------
  def set_order(self, order = None):
    """
    order = 'datn' means order of: `dropout' `architecture', 'transfer function', 'normalisation'
    """
    self.order = order

#-------------------------------------------------------------------------------
  def set_arch_args(self, *args, **kwds):
    """
    *args and **kwds are optional arguments for callable architectures
    """
    self.arch_args = args
    self.arch_kwds = kwds

#-------------------------------------------------------------------------------
  def set_biases(self, bia = None, *bia_args, **bia_kwds):
    """
    bia enables/disables biases and initialisation. Valid inputs are:
    None (default bias settings), False/True, disable/enable biases,
    or Bias initializer (e.g. 'zoi'): use bias with this initialiser
    """
    if bia is None: bia = DEFAULT_BIASES
    self.bia = bia
    self.bia_args = bia_args
    self.bia_kwds = dict(bia_kwds)
    if type(self.bia) is not bool:
      if 'bias_initializer' not in self.bia_kwds:
        self.bia_kwds.update({'bias_initializer': Creation(self.bia)})
      self.bia = True
    self.bia_kwds.update({'use_bias': self.bia})

#-------------------------------------------------------------------------------
  def set_weights(self, wgt = None, *wgt_args, **wgt_kwds):
    """
    Sets initialiser for weights
    wgt = None or 'vs' (variance scaling)
    """
    self.wgt = wgt
    self.wgt_args = wgt_args
    self.wgt_kwds = dict(wgt_kwds)
    if self.wgt is None: return
    if 'kernel' not in self.wgt_kwds:
        self.wgt_kwds.update({'kernel_initializer': Creation(self.wgt)})
    if type(self.wgt) is list or type(self.wgt) is tuple or type(self.wgt) is Creation('var'):
      if 'kernel_transpose' not in self.wgt_kwds:
        if 'transpose' in self.wgt_kwds:
          self.wgt_kwds['kernel_transpose']= self.wgt_kwds.pop('transpose')
        else:
          self.wgt_kwds.update({'kernel_transpose': False})

#-------------------------------------------------------------------------------
  def set_dropout(self, dro = None, *dro_args, **dro_kwds):
    """
    dro = None: No dropout
    dro = 0.: Full dropout (i.e. useless)
    dro = 0.4: dropout with keep probability of 0.6
    """
    if type(dro) is Creation('session'):
      if len(dro_args) == 1 and not(len(dro_kwds)):
        if dro_args[0] is None:
          return
        elif self.dropout_quotient is None:
          return
        op = self.dropout_quotient.assign(dro_args[0])
        return dro.run(op)
      else:
        raise ValueError("Unknown dropout change specification")
    if type(dro) is float and not(len(dro_args)) and not(len(dro_kwds)):
      dro, dro_args = 'var', (dro,)
    if dro is not None and not(len(dro_args)):
      raise ValueError("Unknown dropout specification")

    self.dro = dro
    self.dro_args = dro_args
    self.dro_kwds = dict(dro_kwds)

#-------------------------------------------------------------------------------
  def set_transfn(self, tfn = None, *tfn_args, **tfn_kwds):
    """
    tfn = 'relu': ReLU
    tfn = 'elu': ELU
    other options: 'softmax', and 'sigmoid'
    """
    if tfn is None: tfn = DEFAULT_TRANSFER_FUNCTION
    self.tfn = tfn
    self.tfn_args = tfn_args
    self.tfn_kwds = dict(tfn_kwds)
    self.trans_fn = self.tfn

#-------------------------------------------------------------------------------
  def set_padwin(self, win = None, *win_args, **win_kwds):
    """
    win = 'same' or 'valid'
    """
    self.win = win
    self.win_args = win_args
    self.win_kwds = dict(win_kwds)
    if self.type_arch != 'conv' and self.type_arch != 'pool': return
    if self.win is None: self.win = DEFAULT_PADDING_WINDOW
    if 'padding' not in self.win_kwds:
      self.win_kwds.update({'padding':self.win})

#-------------------------------------------------------------------------------
  def set_kernfn(self, kfn = None, *kfn_args, **kfn_kwds):
    """
    kfn = 'xcorr' or 'sconv' or 'tconv' for convolution layers
    kfn = 'max' or 'avg' for pooling layers
    """
    self.kfn = kfn
    self.kfn_args = kfn_args
    self.kfn_kwds = dict(kfn_kwds)
    if self.kfn is not None: return
    if self.type_arch == 'pool':
      self.kfn = DEFAULT_POOLING_KERNEL_FUNCTION
    elif self.type_arch == 'conv':
      self.kfn = DEFAULT_CONVOLUTION_KERNEL_FUNCTION

#-------------------------------------------------------------------------------
  def set_normal(self, nor = None, *nor_args, **nor_kwds):
    """
    nor = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    self.nor = nor
    self.nor_args = nor_args
    self.nor_kwds = dict(nor_kwds)

#-------------------------------------------------------------------------------
  def set_reguln(self, reg = None, *reg_args, **reg_kwds):
    """
    nor = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    self.reg = reg if type(reg) is not int else 'l'+str(reg)+'_reg'
    self.reg_args = reg_args
    self.reg_kwds = dict(reg_kwds)

#-------------------------------------------------------------------------------
  def __call__(self, inp = None, _called = True):
    """
    inp must be a single tensor.

    Here the graph objects are created.
    """

    # This sets the TensorFlow calls but does not create the graph objects,
    # with the exception of self._call_dropout() which creates scalar
    # objects if required.

    self._call_input(inp)    # specify flatten object if necessary
    if self._inp is None: return self._inp # nothing in, nothing out

    # Default order if necessary
    order = self.order
    if order is None:
      order = 'a' if self.type_arch is 'none' else DEFAULT_STREAM_ORDER

    # Note we don't actually call anything in this for-loop until chain.__call__()
    for _order in order:
      order = _order.lower()
      if order == 'd':
        self._call_dropout()
      elif order == 'a':
        self._call_arch()
      elif order == 't':
        self._call_transfer()
      elif order == 'n':
        self._call_norm()
      else:
        raise ValueError("Unknown order specification: " + str(order))

    # Declare the links as subobjects to have access to their respective parameters
    self.set_subobjects(self._links)

    # Now create the objects
    chain.__call__(self, self._inp, False)

    # Flag the tensor from the architecture-dependent link in the chain
    if self.arch_link is not None: self.arch_out = self.arch_link.ret_out()

    # Collate architectural parameters
    self._setup_params()

    # Collate regularisation parameters
    self._setup_reguln()

    # Set outputs dictionary
    self._setup_outputs()

    self._called = _called

    return self.ret_out()

#-------------------------------------------------------------------------------
  def _call_input(self, inp = None):
    # stream claims no ownership over input
    self._inp = inp
    self._out = None
    if self._inp is None: return self._inp
    if type(self._inp) is not Dtype('tensor'):
      raise TypeError("Input type must be a tensor.")

    # but will claim ownership over any needed flattening operation
    if len(Shape(self._inp)) > 2 and self.type_arch == 'dense':
      self.add_link(Creation('flatten'), name = self.name+"/input_flatten")
    elif len(Shape(self._inp)) == 1 and self.type_arch == 'map2dense':
      self.add_link(Creation('expand_dims'), name = self.name+"/input_expand_dims", axis = -1)
    return inp

#-------------------------------------------------------------------------------
  def _call_dropout(self):
    if self.dro is None or not(len(self.dro_args)): return
    # Here dropout graph scalars are created
    if self.dev is None:
      self.dropout_quotient = Creation(self.dro)(*self.dro_args,
                              name = self.name + "/dropout/quotient", trainable=False)
    else:
      with Device(self.dev):
        self.dropout_quotient = Creation(self.dro)(*self.dro_args,
                                name = self.name + "/dropout/quotient", trainable=False)
    kwds = dict(self.dro_kwds)
    if 'training' not in kwds:
      if self.ist is None:
        raise ValueError("Cannot setup dropout before setting training flag.")
      else:
        kwds.update({'training': self.ist})
    return self.add_link(Creation('dropout'), rate = self.dropout_quotient,
                         name = self.name + "/dropout", **kwds)

#-------------------------------------------------------------------------------
  def _call_arch(self):
    if self.type_adim == 'none':
      return self.ret_out()
    if self.type_adim == 'identity':
      self.add_link(Creation(self.type_adim), name = self.name + "/" + self.type_adim)
      return self.ret_out()

    # Initialise padding/kernel settings
    if self.type_arch == 'conv' or self.type_arch == 'pool':
      if self.win is None: self.set_padwin()
      if self.kfn is None: self.set_kernfn()

    # Initialise biases/weights parameter settings
    maybe_biases = self.type_arch == 'dense' or self.type_arch == 'conv'
    maybe_weights = (maybe_biases or self.type_arch == 'map2dense')
    if maybe_biases:
      if self.bia is None: self.set_biases()
    if maybe_weights:
      if self.wgt is None: self.set_weights()
      if Creation(self.wgt) == Creation('vsi'): # create a custom initialiser
        vsi_kwds = dict(self.wgt_kwds)
        vsi_kwds.pop('kernel_initializer')
        if self.dev is None:
          self.vsi = Creation(self.wgt)(**vsi_kwds)
        else:
          with Device(self.dev):
            self.vsi = Creation(self.wgt)(**vsi_kwds)
        self.wgt_kwds = {'kernel_initializer': self.vsi}

    # Call layers
    type_adim = self.type_adim if self.type_arch != 'callable' else self.type_arch
    kwds = {'name': self.name + "/" + type_adim}
    if self.type_arch == 'callable':
      kwds.update(dict(arch_kwds))
      self.arch_link = self.add_link(self.type_adim, *self.arch_args, **kwds)
    elif self.type_arch == 'dense':
      kwds.update({'units': self.arch})
      kwds.update({'activation': None})
      kwds.update(self.bia_kwds)
      kwds.update(self.wgt_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim), **kwds)
    elif self.type_arch == 'conv':
      kwds.update({'filters': self.arch[0], 'kernel_size': self.arch[1], 'strides': self.arch[2]})
      kwds.update({'activation': None})
      kwds.update(self.bia_kwds)
      kwds.update(self.wgt_kwds)
      kwds.update(self.win_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim, self.kfn), **kwds)
    elif self.type_arch == 'pool':
      kwds.update({'pool_size': self.arch[0], 'strides': self.arch[1]})
      kwds.update(self.win_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim, self.kfn), **kwds)
    elif self.type_arch == 'map2dense':
      arch_key = list(self.arch)[0]
      arch_val = self.arch[arch_key]
      kwds.update(self.wgt_kwds)
      self.arch_link = self.add_link(Creation(self.type_arch), arch_key, arch_val, **kwds)
    else:
      raise ValueError("Unknown architecture type: " + self.type_arch)
    return self.arch_link

#-------------------------------------------------------------------------------
  def _call_transfer(self):
    if self.tfn is None: return self.ret_out()
    kwds = dict(self.tfn_kwds)
    if 'var_scope' not in kwds:
      kwds.update({'var_scope': self.name})
    return self.add_link(Creation(self.tfn), *self.tfn_args, **kwds)

#-------------------------------------------------------------------------------
  def _call_norm(self):
    if self.nor is None: return self.ret_out()
    kwds = dict(self.nor_kwds)
    if Creation(self.nor) == Creation('batch_norm'):
      # Official batch_norm is broken...
      """
      if 'training' not in kwds:
        if self.ist is None:
          raise ValueError("Cannot setup batch_norm before setting training flag.")
        else:
          kwds.update({'training': self.ist})
      if 'name' not in kwds:
        kwds.update({'name': self.name + "/batch_norm"})
      """
      # Official batch_norm is broken...
      if 'is_training' not in kwds:
        if self.ist is None:
          raise ValueError("Cannot setup batch_norm before setting training flag.")
        else:
          kwds.update({'is_training': self.ist})
      if 'update_collections' not in kwds:
        kwds.update({'updates_collections': None})
      if 'scope' not in kwds:
        kwds.update({'scope': self.name + "/batch_norm"})
    self.add_link(Creation(self.nor), *self.nor_args, **kwds)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def _setup_params(self):
    self._params = []
    self._n_params = len(self._params)
    if self.arch_link is None: return self._params
    self._params = self.arch_link._setup_params()
    self._n_params = len(self._params)
    return self._params

#-------------------------------------------------------------------------------
  def _setup_reguln(self):
    self._reguln = []
    if self.reg is None: return self._reguln
    param_reg = list(Param_Reg)[0]
    for param in self._params:
      param_name = list(param)[0]
      if param_reg in param_name:
        self._reguln.append(param.copy())
        self._reguln[-1].update({'reg': self.reg,
                                 'reg_args': self.reg_args,
                                 'reg_kwds': self.reg_kwds,
                                 'name': param_name})
    return self._reguln

#-------------------------------------------------------------------------------
  def _setup_outputs(self):
    # a stream should really have only one output so is more leaf-like here
    self._outputs = []
    self._n_outputs = len(self._outputs)
    if self._out is None: return self._outputs
    self._outputs = [mapping({self.name + "/output": self.ret_out()})]
    self._n_outputs = len(self._outputs)
    return self._outputs

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = stream()
    elif not isinstance(other, stream) and not issubclass(other, stream):
      raise TypeError("Cannot clone to target class " + str(other))
    elif other._links is not None and self.arch != other.arch:
      raise AttributeError("Cannot clone to a target instance with differing architecture")

    # Clone the architecture - does not create links because they are add-appended
    other.set_arch(self.arch)

    # Clone the links (redundant before the setup stage)
    chain.clone(self, other)

    # Now the rest of the specification (tedious, but safe)

    if self.order is not None: other.set_order(self.order)
    if self.type_arch == 'callable': other.set_arch_args(*self.arch_args, **self.arch_kwds)
    if self.ist is not None: other.set_is_training(self.ist)
    if self.bia is not None: other.set_biases(self.bia, *self.bia_args, **self.bia_kwds)
    if self.wgt is not None: other.set_weights(self.wgt, *self.wgt_args, **self.wgt_kwds)
    if self.dro is not None: other.set_dropout(self.dro, *self.dro_args, **self.dro_kwds)
    if self.tfn is not None: other.set_transfn(self.tfn, *self.tfn_args, **self.tfn_kwds)
    if self.win is not None: other.set_padwin(self.win, *self.win_args, **self.win_kwds)
    if self.kfn is not None: other.set_kernfn(self.kfn, *self.kfn_args, **self.kfn_kwds)
    if self.nor is not None: other.set_normal(self.nor, *self.nor_args, **self.nor_kwds)
    if self.reg is not None: other.set_reguln(self.reg, *self.reg_args, **self.reg_kwds)

    # Copy over the summary transfer function

    other.trans_fn = self.trans_fn

    # Rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)

    return other

#-------------------------------------------------------------------------------

