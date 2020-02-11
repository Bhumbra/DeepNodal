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

  # public
  def_name = 'level'
  def_subobject_name = 'stream'
  def_subobject = stream
  arch = None              # architecture
  type_arch = None         # architecture if all stream archectures are the same
  trans_fn = None          # transfer function if identical for all streams
  trans_link = None        # transfer function link
  ipv = None               # Input vergence specification (across streams)
  opv = None               # Output vergence specification (across streams)
  Inp = None               # post_vergence input (a tuple)
  Out = None               # pre_vergence output (a tuple)

  # protected
  _spec_type = tuple        # specification type

#-------------------------------------------------------------------------------
  def __init__(self, name=None, dev=None):
    stem.__init__(self, name, dev)
    self.set_arch()    # defaults to absence of streams
    self.set_ipverge() # sets ipv_args and ipv_kwds
    self.set_opverge() # sets opv_args and opv_kwds

#-------------------------------------------------------------------------------
  def set_arch(self, arch=None):
    """
    Here, arch is a tuple of architecture specifications of a length that
    corresponds to the number of streams within the level, since each
    element within the tuple specifies the stream architecture.
    """
    self.arch = arch
    if self.arch is None:
      self.set_subobjects()
      self.type_arch = None
      return self.type_arch

    if type(self.arch) is not tuple:
      self.arch = (self.arch,)
    set_arch_from_index = 0
    if self._subobjects is None:
      self.set_subobjects(len(self.arch))
    else:
      set_arch_from_index = self._n_subobjects
      for i, subobject in enumerate(self._subobjects):
        if self.arch[i] != subobject.arch:
          set_arch_from_index = i
      self.add_subobjects(len(self.arch) - len(self._subobjects))
    self.type_arch = []
    for i in range(set_arch_from_index, len(self.arch)):
      self._subobjects[i].set_parent() # prevents recursive updates
      self._subobjects[i].set_arch(self.arch[i])
    for i, arch in enumerate(self.arch):
      self._subobjects[i].set_parent(self)
      type_arch = self._subobjects[i].type_arch
      if type_arch not in self.type_arch:
        self.type_arch.append(type_arch)
    self.type_arch = ','.join(self.type_arch)
    if self._parent is not None:
      self._parent.update_arch()
    return self.type_arch

#-------------------------------------------------------------------------------
  def add_arch(self, arch=None):
    output_list = True
    if type(arch) is not tuple:
      arch = (arch,)
      output_list = False
    new_arch = [] if self.arch is None else list(self.arch)
    new_arch += arch
    self.set_arch(tuple(new_arch))
    if output_list:
      return self._subobjects[-len(arch):]
    else:
      return self._subobjects[-1]

#-------------------------------------------------------------------------------
  def update_arch(self): # invoked by stream changes
    arch = []
    for subobject in self._subobjects:
      arch.append(subobject.arch)
    self.arch = tuple(arch)
    if self._parent is not None:
      self._parent.update_arch()
    return self.arch

#-------------------------------------------------------------------------------
  def set_ipverge(self, ipv=None, *ipv_args, **ipv_kwds):
    """
    ipv is a vergence specification that unites inputs expressed as a tuple with a
    length corresponding to that of the number of streams.

    ipv may be True if there is a single stream, specifying all inputs to be verged.

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
      elif self._unit_subobject:
        pass # we'll just have to find out the number of inputs later
      else:
        raise ValueError('Non-unitary stream levels require list-based set_ipverge specifications')
    elif type(self.ipv) is not(self._spec_type):
      raise TypeError("Specification type must be either None, a boolean, or tuple")
    else:
      if len(self.ipv) != self._n_subobjects:
        raise TypeError('Tuple length for set_ipverge specification must match number of streams')
    if 'vergence_fn' not in self.ipv_kwds:
      self.ipv_kwds.update({'vergence_fn': DEFAULT_VERGENCE_FUNCTION})
    if 'axis' not in self.ipv_kwds:
      ax = 0
      if Creation('con') == Creation(self.ipv_kwds['vergence_fn']):
        ax = -1
      self.ipv_kwds.update({'axis': ax})

#-------------------------------------------------------------------------------
  def _set_spec(self, func, spec=None, *args, **kwds): # overloads stem._set_spec
    """
    We overload here because here we 'None' any broadcast specifications to
    'none' architectures.

    To over-rule this effect, use [] ('identity') rather than None ('none').
    """
    if type(spec) is not self._spec_type:
      spec = [spec] * self._n_subobjects
      for i in range(self._n_subobjects):
        if self._subobjects[i].type_adim == 'none':
          spec[i] = None
      spec = self._spec_type(spec)
    return stem._set_spec(self, func, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_is_training(self, spec=None, *args, **kwds):
    """
    spec = is_training must be set to handle some operations (e.g. batch normalisation)
    """
    return self._set_spec(self._subobject.set_is_training, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_order(self, spec=None, *args, **kwds):
    """
    spec = 'datn' means order of: `dropout' `architecture', 'transfer function', 'normalisation'
    """
    return self._set_spec(self._subobject.set_order, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_biases(self, spec=None, *args, **kwds):
    """
    spec enables/disables biases and initialisation. Valid inputs are:
    None (default bias settings), False/True, disable/enable biases,
    or Bias initializer (e.g. 'zoi'): use bias with this initialiser
    """
    return self._set_spec(self._subobject.set_biases, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_weights(self, spec=None, *args, **kwds):
    """
    Sets initialiser for weights
    wgt = None or 'vs' (variance scaling)
    """
    return self._set_spec(self._subobject.set_weights, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_dropout(self, spec=None, *args, **kwds):
    """
    spec = None: No dropout
    spec = 0.: Full dropout (i.e. useless)
    spec = 0.4: dropout with keep probability of 0.6
    """
    return self._set_spec(self._subobject.set_dropout, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_transfn(self, spec=None, *args, **kwds):
    """
    spec = 'relu': ReLU
    spec = 'elu': ELU
    other options: 'softmax', and 'sigmoid'
    """
    argout = self._set_spec(self._subobject.set_transfn, spec, *args, **kwds)
    for i, subobj in enumerate(self[:]):
      if not(i):
        self.trans_fn = self[i].tfn
      elif self[i].tfn != self.trans_fn:
        self.trans_fn = None
    return argout

#-------------------------------------------------------------------------------
  def set_window(self, spec=None, *args, **kwds):
    """
    spec = 'same' or 'valid'
    """
    return self._set_spec(self._subobject.set_window, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_kernfn(self, spec=None, *args, **kwds):
    """
    spec = 'max' or 'avg'
    """
    return self._set_spec(self._subobject.set_kernfn, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_normal(self, spec=None, *args, **kwds):
    """
    spec = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    return self._set_spec(self._subobject.set_normal, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_reguln(self, spec=None, *args, **kwds):
    """
    spec = 'l1_reg' or 'l2_reg' with accompanying keywords (scale) required.
    """
    return self._set_spec(self._subobject.set_reguln, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_opverge(self, opv=None, *opv_args, **opv_kwds):
    """
    opv is a vergence specification that unites outputs expressed as a tuple of
    a length that may differ to that of the number of streams.

    opv may be True, specifying all streams outputs to be verged within the level.

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
      elif self._unit_subobject:
        print('Specification include attempting to verge a single stream')
        self.opv = None
        return
      else:
        self.opv = self._spec_type([list(range(len(self)))])

    if type(self.opv) is self._spec_type:
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
  def clone(self, other=None):
    if other is None:
      other = level()
    elif not isinstance(other, level) and not issubclass(other, level):
      raise TypeError("Cannot clone to target class " + str(other))

    # Clone the architecture - this _will_ create new instances of streams
    if other._subobjects is None:
      other.set_arch(self.arch)
    elif self.arch != other.arch:
      raise AttributeError("Cannot clone to a target instance with differing architecture")

    # Clone the streams
    for self_subobject, other_subobject in zip(self._subobjects, other._subobjects):
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
  def __call__(self, inp=None, _called=True):
    inp = self._call_input(inp)  # does not touch self._subobjects
    if self._inp is None: return self._inp # nothing in, nothing out
    for _inp, subobject in zip(list(inp), self._subobjects):
      subobject.__call__(_inp)
    self.set_called(_called)
    Out = [subobject.ret_out() for subobject in self._subobjects]
    self.pre_trans = None if not self._unit_subobject else self[0].pre_trans
    argout = self._call_output(tuple(Out)) # does not touch self._subobjects
    return argout

#-------------------------------------------------------------------------------
  def _call_input(self, inp=None):
    # inp may either be a level or a tuple of inputs.
    # self.inp -> vergence (if specified) -> self.Inp
    if self.ipv is None: self.set_ipverge()
    self._inp = inp
    self.Inp = self._inp
    self._out = None
    self.Out = None
    if self._inp is None: return self.Inp
    if type(self._inp) is list:
      raise TypeError("Input setup argument must be a tuple, a class, but not a list")
    elif isinstance(self._inp, self._subobject):
      # we have no interest in remembering the identity of input class sources
      inp = tuple([subobject._inp for subobject in self._inp])
      self._inp = inp
    elif type(self._inp) is not tuple: # give the benefit of the doubt here
      self._inp = tuple([self._inp] * self._n_subobjects)
    elif len(self._inp) == 1 and not(self._unit_subobject):
      self._inp = tuple([self._inp[0]] * self._n_subobjects)
    if self.ipv is None: # Attempt an input concentation if input multiple
      if len(self._inp) > 1:
        if self._unit_subobject:
          self.set_ipverge(True, *self.ipv_args, **self.ipv_kwds)
        elif len(self._inp) != self._n_subobjects:
          raise ValueError("Number of input verge specificaitons incommensurate with number of streams")
    if type(self.ipv) is bool:
      if self.ipv:
        self.ipv = self._spec_type([list(range(len(self._inp)))])
    self.Inp = self._inp
    if self.ipv is None: return self.Inp
    if len(self.ipv) != self._n_subobjects:
      raise ValueError("Input vergence specification incommensurate with number of streams")

    # Here if specified we create the graph object(s) for input vergence.
    Inp = [None] * len(self.ipv)
    for i, ipv in enumerate(self.ipv):
      inp = [self._inp[j] for j in ipv]
      func = self.ipv_kwds['vergence_fn']
      kwds = dict(self.ipv_kwds)
      kwds.pop('vergence_fn')
      Inp[i] = Creation(func)(inp, *self.ipv_args,
               name = self.name + "/input_" + func + "vergence_" + str(i), **kwds)
    self.Inp = tuple(Inp)
    return self.Inp

#-------------------------------------------------------------------------------
  def _call_output(self, Out=None):
    # Out is expected to be tuple of size self.n_streams. Optionally it may be
    # a single graph object for single stream levels.
    if self.opv is None: self.set_opverge()
    self.Out = Out
    self._out = self.Out
    if self._out is None: return self.ret_out()
    if type(self.Out) is list:
      raise TypeError("Output setup argument must be a tuple and not a list")
    elif type(self.Out) is not tuple:
      self.Out = tuple(self.Out)
    if len(self.Out) != self._n_subobjects:
      raise ValueError("len(self.Out) must match number of streams")
    self._out = self.Out
    if self.opv is None: return self.ret_out()
    out = [None] * len(self.opv)
    for i, opv in enumerate(self.opv):
      Out = [self.Out[j] for j in opv]
      func = self.opv_kwds['vergence_fn']
      kwds = dict(self.opv_kwds)
      kwds.pop('vergence_fn')
      out[i] = Creation(func)(Out, *self.opv_args,
               name = self.name + "/output_" + func + "vergence_" + str(i), **kwds)
    self._out = tuple(out)
    self._setup_outputs() # concatenate output list of dictionaries
    return self.ret_out()

#-------------------------------------------------------------------------------
  def _setup_reguln(self):
    self._reguln = {'loss': [], 'grad': [], 'vars': []}
    keys = list(self._reguln.keys())
    for subobject in self._subobjects:
      sub_reg = subobject.ret_reguln()
      for key in keys:
        self._reguln[key].extend(sub_reg[key])
    return self._reguln

#-------------------------------------------------------------------------------
  def ret_reguln(self):
    return self._setup_reguln()

#-------------------------------------------------------------------------------
