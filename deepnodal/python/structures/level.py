# Level module for Tensorflow. A level is a structure that contains one or more
# streams in parallel which may verged either at the input ends, output ends,
# both, or not at all and any combination as specified by the self.set_ipverge
# and self.set_opverge. Vergence comprises of the following function types:
# convergence (i.e. concat), divergence (i.e. split), sumvergence (i.e. sum).
# Note that sumvergence is the type often used for residual learning.

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import stem
from deepnodal.python.structures.stream import *

#-------------------------------------------------------------------------------
DEFAULT_VERGENCE_FUNCTION = 'con'

#-------------------------------------------------------------------------------
class level (stem):
  # A level comprises a single stream or multiple streams that may or may not be
  # verged at either end to varying degrees.
  #
  # Therefore each input and output is a tuple.

  def_name = 'level'
  def_subobject_name = 'stream'
  def_subobject = stream
  arch = None              # architecture
  type_arch = None         # architecture if all stream archectures are the same
  arch_out = None          # architecture output for a single stream
  trans_fn = None          # transfer function if identical for all streams
  streams = None           # provides a friendly UI to the list of subobjects
  spec_type = tuple        # specification type
  ipv = None               # Input vergence specification (across streams)
  opv = None               # Output vergence specification (across streams)
  Inp = None               # post_vergence input (a tuple)
  Out = None               # pre_coalesence output (a tuple)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)
    self.set_arch()    # defaults to an identity
    self.set_ipverge() # sets ipv_args and ipv_kwds
    self.set_opverge() # sets ipv_args and ipv_kwds
    self.setup()

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None): 
    """
    Here, arch is a tuple of architecture specifications of a length that
    corresponds to the number of streams within the level, since each
    element within the tuple specifies the stream architecture.
    """
    self.arch = arch
    if self.arch is None: return
    if type(self.arch) is not tuple:
      if type(self.arch) is int or type(self.arch) is list:
        self.arch = (self.arch,)
      elif type(self.arch) is not tuple:
        raise TypeError("Unknown level architecture: " + str(self.arch))
    self.set_subobjects(len(self.arch))
    self.type_arch = []
    for i, arch in enumerate(self.arch):
      self.subobjects[i].set_arch(arch)
      type_arch = self.subobjects[i].type_arch
      if type_arch not in self.type_arch:
        self.type_arch.append(type_arch)
    self.streams = self.subobjects
    self.type_arch = ','.join(self.type_arch)
    return self.type_arch

#-------------------------------------------------------------------------------
  def set_ipverge(self, ipv = None, *ipv_args, **ipv_kwds):
    """
    ipv is a vergence specification that unites inputs expressed as a tuple with a
    length corresponding to that of the number of streams.

    ipv may be True if there is a single stream, specifying all inputs to be coalesced.

    By default, convergence ('con') is assumed, but under `kwds', 'vergence_fn'
    may be 'sum'.

    The responsibility of dimension commensuration is with the network designer.

    """
    self.ipv = ipv
    self.ipv_args = ipv_args
    self.ipv_kwds = dict(ipv_kwds)
    if self.ipv is None: return
    if type(self.ipv) is bool:
      if not(self.ipv):
        self.ipv = None
        return
      elif self.unit_subobject:
        pass # we'll just have to find out the number of inputs later
      else:
        raise ValueError('Non-unitary stream levels require list-based set_ipverge specifications')
    elif type(self.ipv) is not(self.spec_type):
      raise TypeError("Specification type must be either None, a boolean, or tuple")
    else:
      if len(self.ipv) != self.n_subobjects:
        raise TypeError('Tuple length for set_ipverge specification must match number of streams')
    if 'vergence_fn' not in self.ipv_kwds:
      self.ipv_kwds.update({'vergence_fn': DEFAULT_VERGENCE_FUNCTION})
    if 'axis' not in self.ipv_kwds:
      ax = 0
      if Creation('con') == Creation(self.ipv_kwds['vergence_fn']):
        ax = -1
      self.ipv_kwds.update({'axis': ax})

#-------------------------------------------------------------------------------
  def set_spec(self, func, spec = None, *args, **kwds): # overloads stem.broadcast
    """
    We overload here because here we 'None' any broadcast specifications to
    identity architectures.

    There's nothing to stop designers from over-ruling this by providing
    full list specifications but they would really want to.
    """
    #if type(spec) is not list: spec = [spec] * self.n_subobjects
    if type(spec) is not self.spec_type:
      spec = [spec] * self.n_subobjects
      for i in range(self.n_subobjects):
        if self.subobjects[i].type_adim == 'identity':
          spec[i] = None
      spec = self.spec_type(spec)
    return stem.set_spec(self, func, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_is_training(self, spec = None, *args, **kwds):
    """
    spec = is_training must be set to handle some operations (e.g. batch normalisation)
    """
    return self.set_spec(self.subobject.set_is_training, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_order(self, spec = None, *args, **kwds):
    """
    spec = 'datn' means order of: `dropout' `architecture', 'transfer function', 'normalisation'
    """
    return self.set_spec(self.subobject.set_order, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_biases(self, spec = None, *args, **kwds):
    """
    spec enables/disables biases and initialisation. Valid inputs are:
    None (default bias settings), False/True, disable/enable biases,
    or Bias initializer (e.g. 'zoi'): use bias with this initialiser
    """
    return self.set_spec(self.subobject.set_biases, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_weights(self, spec = None, *args, **kwds):
    """
    Sets initialiser for weights
    wgt = None or 'vs' (variance scaling)
    """
    return self.set_spec(self.subobject.set_weights, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_dropout(self, spec = None, *args, **kwds):
    """
    spec = None: No dropout
    spec = 0.: Full dropout (i.e. useless)
    spec = 0.4: dropout with keep probability of 0.6
    """ 
    return self.set_spec(self.subobject.set_dropout, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_transfn(self, spec = None, *args, **kwds):
    """
    spec = 'relu': ReLU
    spec = 'elu': ELU
    other options: 'softmax', and 'sigmoid'
    """
    argout = self.set_spec(self.subobject.set_transfn, spec, *args, **kwds)
    for i, subobj in enumerate(self.subobjects):
      if not(i):
        self.trans_fn = self.subobjects[i].tfn
      elif self.subobjects[i].tfn != self.trans_fn:
        self.trans_fn = None
    return argout

#-------------------------------------------------------------------------------
  def set_padwin(self, spec = None, *args, **kwds):
    """
    spec = 'same' or 'valid'
    """
    return self.set_spec(self.subobject.set_padwin, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_kernfn(self, spec = None, *args, **kwds):
    """
    spec = 'max' or 'avg'
    """
    return self.set_spec(self.subobject.set_kernfn, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_normal(self, spec = None, *args, **kwds):
    """
    spec = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    return self.set_spec(self.subobject.set_normal, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_opverge(self, opv = None, *opv_args, **opv_kwds):
    """
    opv is a vergence specification that unites outputs expressed as a tuple of
    a length that may differ to that of the number of streams.

    opv may be True, specifying all streams outputs to be coalesced within the level.

    ...or opv may be a list of lists where the outer dimension is the number of
    vergences and inner dimension for each is the index of streams to verge.

    By default, concatenation ('con') is assumed, but under `kwds', 'vergence_fn'
    may be 'sum'.

    The responsibility of dimension commensuration is with the network designer.

    """
    self.opv = opv
    self.opv_args = opv_args
    self.opv_kwds = dict(opv_kwds)
    if self.opv is None: return

    # There is no reason why we can't complete and check the specification now

    # First check/complete boolean specification
    if type(self.opv) is bool:
      if not(self.opv):
        self.opv = None
        return
      elif self.unit_subobject:
        print('Specification include attempting to coalesce a single stream')
        self.opv = None
        return
      else:
        self.opv = self.spec_type([list(range(self.n_subobjects))])

    if type(self.opv) is self.spec_type:
      if not(len(self.opv)):
        self.opv = None
        return
    else:
      raise TypeError('Specification must either be boolean, None, or a tuple of lists')
    if 'vergence_fn' not in self.opv_kwds:
      self.opv_kwds.update({'vergence_fn': DEFAULT_VERGENCE_FUNCTION})
    if 'axis' not in self.opv_kwds:
      ax = 0
      if Creation('con') == Creation(self.opv_kwds['vergence_fn']):
        ax = -1
      self.opv_kwds.update({'axis': ax})

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = level()
    elif not isinstance(other, level) and not issubclass(other, level):
      raise TypeError("Cannot clone to target class " + str(other))

    # Clone the architecture - this _will_ create new instances of streams
    if other.subobjects is None:
      other.set_arch(self.arch)
    elif self.arch != other.arch:
      raise AttributeError("Cannot clone to a target instance with differing architecture")

    # Clone the streams
    for self_subobject, other_subobject in zip(self.subobjects, other.subobjects):
      self_subobject.clone(other_subobject)

    # Now the rest of the level-rank specifications
    if self.ipv is not None: other.set_ipverge(self.ipv, *self.ipv_args, **self.ipv_kwds)
    if self.opv is not None: other.set_opverge(self.opv, *self.opv_args, **self.opv_kwds)

    # Copy over the summary transfer function

    other.trans_fn = self.trans_fn

    # Rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)

    return other

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    inp = self._setup_input(inp)  # does not touch self.subobjects
    if self.inp is None: return self.inp # nothing in, nothing out
    for _inp, subobject in zip(list(inp), self.subobjects):
      subobject.setup(_inp)
    Out = [subobject.ret_out() for subobject in self.subobjects]
    self.arch_out = None if not self.unit_subobject else self.subobjects[0].arch_out
    return self._setup_output(tuple(Out)) # does not touch self.subobjects

#-------------------------------------------------------------------------------
  def _setup_input(self, inp = None):
    # inp may either be a level or a tuple of inputs.
    # self.inp -> vergence (if specified) -> self.Inp
    if self.ipv is None: self.set_ipverge()
    self.inp = inp
    self.Inp = self.inp
    self.out = None
    self.Out = None
    if self.inp is None: return self.Inp
    if type(self.inp) is list:
      raise TypeError("Input setup argument must be a tuple, a class, but not a list")
    elif isinstance(self.inp, self.subobject):
      # we have no interest in remembering the identity of input class sources
      inp = tuple([subobject.inp for subobject in self.inp])
      self.inp = inp
    elif type(self.inp) is not tuple: # give the benefit of the doubt here
      self.inp = tuple([self.inp] * self.n_subobjects)
    elif len(self.inp) == 1 and not(self.unit_subobject):
      self.inp = tuple([self.inp[0]] * self.n_subobjects)
    if self.ipv is None: # Attempt an input concentation if input multiple
      if len(self.inp) > 1:
        if self.unit_subobject:
          self.set_ipverge(True, *self.ipv_args, **self.ipv_kwds)
        elif len(self.inp) != self.n_subobjects:
          raise ValueError("Number of input verge specificaitons incommensurate with number of streams")
    if type(self.ipv) is bool:
      if self.ipv:
        self.ipv = self.spec_type([list(range(len(self.inp)))])
    self.Inp = self.inp
    if self.ipv is None: return self.Inp
    if len(self.ipv) != self.n_subobjects:
      raise ValueError("Input vergence specification incommensurate with number of streams")

    # Here if specified we create the graph object(s) for input vergence.
    Inp = [None] * len(self.ipv)
    for i, ipv in enumerate(self.ipv):
      inp = [self.inp[j] for j in ipv]
      func = self.ipv_kwds['vergence_fn']
      kwds = dict(self.ipv_kwds)
      kwds.pop('vergence_fn')
      Inp[i] = Creation(func)(inp, *self.ipv_args,
               name = self.name + "/input_" + func + "vergence_" + str(i), **kwds)
    self.Inp = tuple(Inp)
    return self.Inp

#-------------------------------------------------------------------------------
  def _setup_output(self, Out = None):
    # Out is expected to be tuple of size self.n_streams. Optionally it may be
    # a single graph object for single stream levels.
    if self.opv is None: self.set_opverge()
    self.Out = Out
    self.out = self.Out
    if self.out is None: return self.ret_out()
    if type(self.Out) is list:
      raise TypeError("Output setup argument must be a tuple and not a list")
    elif type(self.Out) is not tuple:
      self.Out = tuple(self.Out)
    if len(self.Out) != self.n_subobjects:
      raise ValueError("len(self.Out) must match number of streams")
    self.out = self.Out
    if self.opv is None: return self.ret_out()
    out = [None] * len(self.opv)
    for i, opv in enumerate(self.opv):
      Out = [self.Out[j] for j in opv]
      func = self.opv_kwds['vergence_fn']
      kwds = dict(self.opv_kwds)
      kwds.pop('vergence_fn')
      out[i] = Creation(func)(Out, *self.opv_args,
               name = self.name + "/output_" + func + "vergence_" + str(i), **kwds)
    self.out = tuple(out)
    self.setup_outputs() # concatenate output list of dictionaries
    return self.ret_out()

#-------------------------------------------------------------------------------

