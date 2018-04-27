# Level module for Tensorflow. A level is a structure that contains one or more
# streams in parallel which may coalesced either at the input ends, output ends,
# both, or not at all and any combination as specified by the self.set_ipcoal
# and self.set_opcoal.

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import stem
from deepnodal.python.structures.stream import *

#-------------------------------------------------------------------------------
DEFAULT_COALESCENCE_FUNCTION = 'con'

#-------------------------------------------------------------------------------
class level (stem):
  # A level comprises a single stream or multiple streams that may or may not be
  # coalesced at either end to varying degrees.
  #
  # Therefore each input and output is a tuple.

  def_name = 'level'
  def_subobject_name = 'stream'
  def_subobject = stream
  arch = None              # architecture
  type_arch = None         # architecture if all stream archectures are the same
  arch_out = None          # architecture output for a single stream
  trans_fn = None          # transfer function if identical for all streams
  ipc = None               # Input coalescence specification (across streams)
  opc = None               # Output coalescence specification (across streams)
  Inp = None               # post_coalescence input (a tuple)
  Out = None               # pre_coalesence output (a tuple)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)
    self.set_arch() # defaults to an identity
    self.setup()

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None):
    """
    Here, arch is a tuple of architecture specifications of a length that
    corresponds to the number of streams within the level, since each
    element within the tuple specifies the stream architecture.
    """
    self.arch = arch
    if type(self.arch) is not tuple:
      if self.arch is None or type(self.arch) is int or type(self.arch) is list:
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
    self.type_arch = ','.join(self.type_arch)
    return self.type_arch

#-------------------------------------------------------------------------------
  def set_ipcoal(self, ipc = None, *ipc_args, **ipc_kwds):
    """
    ipc is a coalescence specification that unites inputs so that their numbers must
    correspond to that of the number of streams.

    ipc may be True if there is a single stream, specifying all inputs to be coalesced.

    By default, concatenation ('con') is assumed, but under `kwds', 'coalescence_fn'
    may be 'sum'.

    The responsibility of dimension commensuration is with the network designer.

    """
    self.ipc = ipc
    self.ipc_args = ipc_args
    self.ipc_kwds = dict(ipc_kwds)
    if self.ipc is None: return
    if type(self.ipc) is bool:
      if not(self.ipc):
        self.ipc = None
        return
      elif self.unit_subobject:
        pass # we'll just have to find out the number of inputs later
      else:
        raise ValueError('Non-unitary stream levels require list-based set_ipcoal specifications')
    elif type(self.ipc) is list:
      if not(len(self.ipc)):
        self.ipc = None
        return
      elif type(self.ipc[0]) is not list:
        self.ipc = [self.ipc]
      if len(self.ipc) != self.n_subobjects:
        raise TypeError('List length for set_ipcoal specification must match number of streams')
    if not(len(self.ipc_kwds)):
      self.ipc_kwds.update({'coalescence_fn': DEFAULT_COALESCENCE_FUNCTION})
    if 'axis' not in self.ipc_kwds:
      self.ipc_kwds.update({'axis': -1})

#-------------------------------------------------------------------------------
  def set_is_training(self, spec = None, *args, **kwds):
    """
    spec = is_training must be set to handle some operations (e.g. batch normalisation)
    """
    return self.broadcast(self.subobject.set_is_training, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_order(self, spec = None, *args, **kwds):
    """
    spec = 'datn' means order of: `dropout' `architecture', 'transfer function', 'normalisation'
    """
    return self.broadcast(self.subobject.set_order, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_usebias(self, spec = None, *args, **kwds):
    """
    spec is a boolean flag set to whether to usebias
    """
    return self.broadcast(self.subobject.set_usebias, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_dropout(self, spec = None, *args, **kwds):
    """
    spec = None: No dropout
    spec = 0.: Full dropout (i.e. useless)
    spec = 0.4: dropout with keep probability of 0.6
    """ 
    return self.broadcast(self.subobject.set_dropout, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_transfn(self, spec = None, *args, **kwds):
    """
    spec = 'relu': ReLU
    spec = 'elu': ELU
    other options: 'softmax', and 'sigmoid'
    """
    argout = self.broadcast(self.subobject.set_transfn, spec, *args, **kwds)
    for i, subobj in enumerate(self.subobjects):
      if not(i):
        self.trans_fn = self.subobjects[i].tfn
      elif self.subobjects[i].tfn != self.trans_fn:
        self.trans_fn = None

#-------------------------------------------------------------------------------
  def set_padwin(self, spec = None, *args, **kwds):
    """
    spec = 'same' or 'valid'
    """
    return self.broadcast(self.subobject.set_padwin, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_poolfn(self, spec = None, *args, **kwds):
    """
    spec = 'max' or 'avg'
    """
    return self.broadcast(self.subobject.set_poolfn, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_parinit(self, pin = None, *pin_args, **pin_kwds):
    """
    spec = 'vsi' (variance scale initialiser) and/or 'zoi' (zero offset initialiser)
    """
    return self.broadcast(self.subobject.set_parinit, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_normal(self, nor = None, *nor_args, **nor_kwds):
    """
    spec = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    return self.broadcast(self.subobject.set_normal, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_opcoal(self, opc = None, *opc_args, **opc_kwds):
    """
    opc is a coalescence specification that unites outputs so that their numbers may
    differ to that of the number of streams.

    opc may be True, specifying all streams outputs to be coalesced within the level.

    By default, concatenation ('con') is assumed, but under `kwds', 'coalescence_fn'
    may be 'sum'.

    The responsibility of dimension commensuration is with the network designer.

    """
    self.opc = opc
    self.opc_args = opc_args
    self.opc_kwds = dict(opc_kwds)
    if self.opc is None: return
    if type(self.opc) is bool:
      if not(self.opc):
        self.opc = None
        return
      elif self.unit_subobject:
        print('Specification include attempting to coalesce a single stream')
        pass
      else:
        self.opc = list(range(self.n_subobjects)) 
    if type(self.ipc) is list:
      if not(len(self.opc)):
        self.opc = None
        return
      elif type(self.opc[0]) is not list:
        self.opc = [self.opc]
    else:
      raise TypeError('Specification must either be boolean or a list of lists')
    if not(len(self.opc_kwds)):
      self.opc_kwds.update({'coalescence_fn': DEFAULT_COALESCENCE_FUNCTION})
    if 'axis' not in self.opc_kwds:
      self.opc_kwds.update({'axis': -1})

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = level()
    elif not isinstance(other, level) and not issubclass(other, level):
      raise TypeError("Cannot clone level to class " + str(other))
    elif other.subobjects is not None:
      raise AttributeError("Cannot clone to a level instance with pre-existing subobjects")

    # Clone the architecture - this _will_ create new instances of streams
    other.set_arch(self.arch)

    # Clone the streams
    for self_subobjects, other_subobject in zip(self.subobjects, other.subobjects):
      self_subobject.clone(other_subobject)

    # Now the rest of the specification (tedious, but safe)

    if self.ipc is not None: other.set_ipcoal(self.ipc, *self.ipc_args, **self.ipc_kwds)
    if self.is_training is None: other.set_is_training(self.is_training)
    if self.order is not None: other.set_order(self.order)
    if self.ubi is not None: other.set_usebias(self.ubi, *self.ubi_args, **self.ubi_kwds)
    if self.dro is not None: other.set_dropout(self.dro, *self.dro_args, **self.dro_kwds)
    if self.tfn is not None: other.set_transfn(self.tfn, *self.tfn_args, **self.tfn_kwds)
    if self.win is not None: other.set_padwin(self.win, *self.win_args, **self.win_kwds)
    if self.pfn is not None: other.set_poolfn(self.pfn, *self.pfn_args, **self.pfn_kwds)
    if self.pin is not None: other.set_parinit(self.pin, *self.pin_args, **self.pin_kwds)
    if self.nor is not None: other.set_normal(self.nor, *self.nor_args, **self.nor_kwds)
    if self.opc is not None: other.set_opcoal(self.opc, *self.opc_args, **self.opc_kwds)

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
    # self.inp -> coalescence (if specified) -> self.Inp
    if self.ipc is None: self.set_ipcoal()
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
    if self.ipc is None: # Attempt an input concentation if input multiple
      if len(self.inp) > 1 and self.unit_subobject: # with unit_stream
        self.set_ipcoal(True)
    if type(self.ipc) is bool:
      if self.ipc:
        self.ipc = list(range(len(self.inp))),

    self.Inp = self.inp
    if self.ipc is None: return self.Inp

    if len(self.ipc) != self.n_subobjects:
      raise ValueError("Input coalescence specification incommensurate with number of streams")

    # Here if specified we create the graph object(s) for input coalescence.
    Inp = [None] * len(self.ipc)
    for i, ipc in enumerate(self.ipc):
      inp = [self.inp[j] for j in ipc]
      func = self.ipc_kwds['coalescence_fn']
      kwds = dict(self.ipc_kwds)
      kwds.pop('coalescence_fn')
      Inp[i] = Creation(func)(inp, *self.ipc_args,
               name = self.name + "/input_" + func + "_" + str(i), **kwds)
    self.Inp = tuple(Inp)
    return self.Inp

#-------------------------------------------------------------------------------
  def _setup_output(self, Out = None):
    # Out is expected to be tuple of size self.n_streams. Optionally it may be
    # a single graph object for single stream levels.
    if self.opc is None: self.set_opcoal()
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
    if self.opc is None: return self.ret_out()
    out = [None] * len(self.opc)
    for i, opc in enumerate(self.opc):
      Out = [self.Out[j] for j in opc]
      func = self.opc_kwds['coalescence_fn']
      kwds = self.opc_kwds.pop('coalescence_fn')
      out[i] = Creation(func)(inp, *self.opc_args,
               name = self.name + "/output_" + func + "_" + str(i), **kwds)
    self.out = tuple(out)
    self.setup_outputs() # concatenate output list of dictionaries
    return self.ret_out()


#-------------------------------------------------------------------------------

