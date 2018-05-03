# Stream module for TensorFlow which abstracts out the unintuitive TensorFlow syntax
# to something more consistent.

# Gary Bhumbra

#-------------------------------------------------------------------------------
import numpy as np
from deepnodal.python.structures.chain import *

#-------------------------------------------------------------------------------
DEFAULT_STREAM_ORDER = 'datn' # over-ruled to only 'a' for identity architectures.
DEFAULT_USE_BIAS = True
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

  There may be a maximum of 5 rather than 4 links in the chain since input flattening
  may precede the above list.

  self.set_`function' sets parameters involved with the stream design but does not
  create any TensorFlow objects.

  self._setup_`function' generates TensorFlow objects but should only be called via
  self.setup(inp), which should be inkoved by a function-derived class.
  """

  def_name = 'stream'
  arch = None         # architecture
  arch_link = None    # architecture link
  arch_out = None     # architecture output (i.e. raw weighted sum or pool result)
  type_arch = None    # architecture type without dimension
  type_adim = None    # architecture type with dimension
  order = None        # string denoting order of operation (default 'dotn')
  ist = None          # is_training flag
  ubi = None          # Use bias 
  dro = None          # Dropout
  tfn = None          # Transfer function
  win = None          # Padding window
  kfn = None          # Kernel function (for convolution and pooling layers)
  pin = None          # Parameter initialisation
  nor = None          # Normalisation
  trans_fn = None     # Identical to tfn
  dro_link = None     # references to dropout_link
  dropout_quotient = None

#-------------------------------------------------------------------------------
  def __init__(self, name = 'stream', dev = None):
    chain.__init__(self, name, dev)
    self.set_arch() # defaults to an identity
    self.setup()

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None):
    """
    arch = None or []:  identity
    arch = integer:     dense
    arch = list of 2:   pool
    arch = list of 3:   conv
    """
    # Note links are not created here because they are added sequentially at
    # the stream.setup(inp) stage.

    self.arch = arch # arch = None is the default signifying an identity
    self.arch_link = None
    self.arch_out = None
    self.type_arch = None
    self.type_adim = None
    if self.arch is None:
      self.type_arch = 'identity'
      self.type_adim = self.type_arch
    elif type(self.arch) is int:
      self.type_arch = 'dense'
      self.type_adim = self.type_arch
    elif type(self.arch) is not list:
      raise ValueError("Unknown architecture specification")
    else:
      if len(arch) == 0:
        self.type_arch = 'identity'
        self.type_adim = self.type_arch
      if len(self.arch) == 1:
        raise ValueError("List length of 1 reserved (for recurrent architectures)")
      elif len(self.arch) > 3:
        raise ValueError("Unknown architecture specification")
      else:
        if type(self.arch[1]) is not list:
          raise ValueError("Unknown architecture specification")
        self.type_arch = 'pool' if type(self.arch[0]) is list else 'conv'
        self.type_adim = self.type_arch + str(len(self.arch[1])) + "d"
        if self.type_arch == 'conv' and len(self.arch) == 2: # default to unit stride
          self.arch = [self.arch[0], self.arch[1], [1]*len(self.arch[1])]

    # Default bare essentials
    self.set_order()
    self.set_usebias()
    self.set_transfn()
    self.set_padwin()
    self.set_kernfn()
    return self.type_arch

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
    if order is None:
      order = 'a' if self.type_arch is 'identity' else DEFAULT_STREAM_ORDER
    self.order = order

#-------------------------------------------------------------------------------
  def set_usebias(self, ubi = None, *ubi_args, **ubi_kwds):
    """
    ubi is a boolean flag set to whether to usebias
    """
    if ubi is None: ubi = DEFAULT_USE_BIAS
    self.ubi = ubi
    self.ubi_args = ubi_args
    self.ubi_kwds = dict(ubi_kwds)
    if not(len(ubi_kwds)):
      ubi_kwds = {'use_bias': self.ubi}

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

#-------------------------------------------------------------------------------
  def set_kernfn(self, kfn = None, *kfn_args, **kfn_kwds):
    """
    kfn = 'max' or 'avg'
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
  def set_parinit(self, pin = None, *pin_args, **pin_kwds):
    """
    pin = 'vsi' (variance scale initialiser) and/or 'zoi' (zero offset initialiser)
    """
    self.pin = pin if type(pin) is list else [pin]
    self.pin = [Creation(_pin) for _pin in self.pin]
    self.pin_args = pin_args
    self.pin_kwds = dict(pin_kwds)
    call_vsi = Creation('vsi')
    if call_vsi in self.pin:
      if 'weights_initializer' not in self.pin_kwds:
        self.pin_kwds.update({'weights_initializer', call_vsi})
    call_zoi = Creation('zoi')
    if call_zoi in self.pin:
      if 'bias_initializer' not in self.pin_kwds:
        self.pin_kwds.update({'bias_initializer', call_zoi})

#-------------------------------------------------------------------------------
  def set_normal(self, nor = None, *nor_args, **nor_kwds):
    """
    nor = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    self.nor = nor
    self.nor_args = nor_args
    self.nor_kwds = dict(nor_kwds)

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    """
    inp must be a single tensor.

    Here the graph objects are created.
    """

    # This sets the TensorFlow calls but does not create the graph objects,
    # with the exception of self._setup_dropout() which creates scalar
    # objects if required.

    self._setup_input(inp)    # specify flatten object if necessary
    if self.inp is None: return self.inp # nothing in, nothing out
    for _order in self.order:
      order = _order.lower()
      if order == 'd':
        self._setup_dropout()
      elif order == 'a':
        self._setup_arch()
      elif order == 't':
        self._setup_transfer()
      elif order == 'n':
        self._setup_norm()
      else:
        raise ValueError("Unknown order specification: " + str(order))

    # Declare the links as subobjects to have access to their respective parameters
    self.set_subobjects(self.links)

    # Now create the objects
    chain.setup(self, self.inp)

    # Flag the tensor from the architecture-dependent link in the chain
    if self.arch_link is not None: self.arch_out = self.arch_link.ret_out()

    # Collate architectural parameters
    self.setup_params()

    # Set outputs dictionary
    self.setup_outputs()

    return self.ret_out()

#-------------------------------------------------------------------------------
  def _setup_input(self, inp = None):
    # stream claims no ownership over input
    self.inp = inp
    self.out = None
    if self.inp is None: return self.inp
    if type(self.inp) is not Dtype('tensor'):
      raise TypeError("Input type must be a tensor.")

    # but will claim ownership over any needed flattening operation
    if len(Shape(self.inp)) > 2 and self.type_arch == 'dense':
      self.add_link(Creation('flatten'), name = self.name+"/input_flatten")
    return inp

#-------------------------------------------------------------------------------
  def _setup_dropout(self):
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
  def _setup_arch(self):
    if self.type_adim == 'identity':
      self.add_link(Creation(self.type_adim), name = self.name + "/" + self.type_adim)
      return self.ret_out()
    if self.type_arch == 'conv' or self.type_arch == 'pool':
      if self.win is None: self.set_padwin()
      if self.kfn is None: self.set_kernfn()
    if self.type_arch == 'dense' or self.type_arch == 'conv':
      if self.ubi is None: self.set_usebias()
      if self.pin is None: self.set_parinit()
    kwds = {'name': self.name + "/" + self.type_adim}
    if self.type_arch == 'dense':
      kwds.update({'units': self.arch})
      kwds.update({'activation': None})
      kwds.update(self.ubi_kwds)
      kwds.update(self.pin_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim), **kwds)
    elif self.type_arch == 'conv':
      kwds.update({'filters': self.arch[0], 'kernel_size': self.arch[1], 'strides': self.arch[2]})
      kwds.update({'activation': None})
      kwds.update(self.ubi_kwds)
      kwds.update(self.pin_kwds)
      kwds.update(self.win_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim, self.kfn), **kwds)
    elif self.type_arch == 'pool':
      kwds.update({'pool_size': self.arch[0], 'strides': self.arch[1]})
      kwds.update(self.win_kwds)
      self.arch_link = self.add_link(Creation(self.type_adim, self.kfn), **kwds)
    else:
      raise ValueError("Unknown architecture type: " + self.type_arch)
    return self.arch_link

#-------------------------------------------------------------------------------
  def _setup_transfer(self):
    if self.tfn is None: return self.ret_out()
    kwds = dict(self.tfn_kwds)
    if 'var_scope' not in kwds:
      kwds.update({'var_scope': self.name})
    return self.add_link(Creation(self.tfn), *self.tfn_args, **kwds)

#-------------------------------------------------------------------------------
  def _setup_norm(self):
    if self.nor is None: return self.ret_out()
    kwds = dict(self.nor_kwds)
    if Creation(self.nor) == Creation('batch_norm'):
      if 'training' not in kwds:
        if self.ist is None:
          raise ValueError("Cannot setup batch_norm before setting training flag.")
        else:
          kwds.update({'training': self.ist})
      if 'name' not in kwds:
        kwds.update({'name': self.name + "/batch_norm"})
    self.add_link(Creation(self.nor), *self.nor_args, **kwds)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def setup_params(self):
    self.params = []
    self.n_params = len(self.params)
    if self.arch_link is None: return self.params
    self.params = self.arch_link.setup_params()
    self.n_params = len(self.params)
    return self.params

#-------------------------------------------------------------------------------
  def setup_outputs(self): 
    # a stream should really have only one output so is more leaf-like here
    self.outputs = []
    self.n_outputs = len(self.outputs)
    if self.out is None: return self.outputs
    self.outputs = [{self.name + "/output": self.out}]
    self.n_outputs = len(self.outputs)
    return self.outputs

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = stream()
    elif not isinstance(other, stream) and not issubclass(other, stream):
      raise TypeError("Cannot clone to target class " + str(other))
    elif other.links is not None and self.arch != other.arch:
      raise AttributeError("Cannot clone to a target instance with differing architecture")

    # Clone the architecture - does not create links because they are add-appended
    other.set_arch(self.arch)

    # Clone the links (redundant before the setup stage)
    chain.clone(self, other)

    # Now the rest of the specification (tedious, but safe)

    if self.order is not None: other.set_order(self.order)
    if self.ist is not None: other.set_is_training(self.ist)
    if self.ubi is not None: other.set_usebias(self.ubi, *self.ubi_args, **self.ubi_kwds)
    if self.dro is not None: other.set_dropout(self.dro, *self.dro_args, **self.dro_kwds)
    if self.tfn is not None: other.set_transfn(self.tfn, *self.tfn_args, **self.tfn_kwds)
    if self.win is not None: other.set_padwin(self.win, *self.win_args, **self.win_kwds)
    if self.kfn is not None: other.set_kernfn(self.kfn, *self.kfn_args, **self.kfn_kwds)
    if self.pin is not None: other.set_parinit(self.pin, *self.pin_args, **self.pin_kwds)
    if self.nor is not None: other.set_normal(self.nor, *self.nor_args, **self.nor_kwds)

    # Copy over the summary transfer function

    other.trans_fn = self.trans_fn

    # Rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)

    return other

#-------------------------------------------------------------------------------

