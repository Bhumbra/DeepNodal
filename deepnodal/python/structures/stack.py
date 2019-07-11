"""
Stack module for Tensorflow. A stack is a structure that contains one or more
levels in series with additional support for uninterrupted skip connections.
If skip-connections with intervening computations are required, then this can
be created using multiple streams across levels using (None, arch_spec) arch
specifications where 'None' denotes only an `identity'.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import stem
from deepnodal.python.structures.level import *

#-------------------------------------------------------------------------------
DEFAULT_SKIP_END = 'out'

#-------------------------------------------------------------------------------
class stack (stem):
  # A stack comprises a single level or multiple levels. Stacks can be connected
  # to one or more stacks at both input and output ends. Therefore in principle,
  # multiple stacks should be all that is necessary to design any network.

  # public
  def_name = 'stack'
  def_subobject_name = 'level'
  def_subobject = level
  arch = None             # architecture
  type_arch = None        # level architectures if all stream types are the same
  trans_fn = None         # transfer function of last level if identical for all streams
  arch_out = None         # archecture output of last level if unit_stream
  pre_trans = None        # pre-trasnfer function of last evel
  scv = None              # Skip vergence specification (across levels)

  # protected
  _spec_type = list        # specification type
#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)
    self.set_arch()   # defaults to an identity
    self.set_skipcv() # sets scv_args and scv_kwds

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None):
    self.arch = arch
    if self.arch is None: return
    if type(self.arch) is not list:
      if type(self.arch) is tuple:
        self.arch = [self.arch]
      elif type(self.arch) is int: # ! sloppy but barely acceptable
        self.arch = [(self.arch,)]
      else:
        raise TypeError("Unknown level architecture: " + str(self.arch))
    set_arch_from_index = 0
    if self._subobjects is None:
      self.set_subobjects(len(self.arch))
    else:
      set_arch_from_index = self._n_subobjects
      for i, subobject in enumerate(self._subobjects):
        if self.arch[i] != subobject.arch:
          set_arch_from_index = i
      self.add_subobjects(len(self.arch) - len(self._subobjects))
    for i in range(set_arch_from_index, len(self.arch)):
      self._subobjects[i].set_parent() # prevents recursive updates
      self._subobjects[i].set_arch(self.arch[i])
    self.type_arch = [None] * self._n_subobjects
    for i, arch in enumerate(self.arch):
      self._subobjects[i].set_parent(self)
      self.type_arch[i] = self._subobjects[i].type_arch
    return self.type_arch

#-------------------------------------------------------------------------------
  def add_arch(self, arch = None):
    arch_tuple = False
    if type(arch) is list:
      for _arch in arch:
        enlist = False
        arch_tuple = True
        if type(_arch) is not tuple:
          if arch_tuple is False:
            raise TypeError("Level specifications exhibit inconsistent datatypes")
          arch_tuple = False
        elif type(_arch) is int:
          enlist = True
    else:
      enlist = True
    if enlist:
      arch = [arch]
    new_arch = [] if self.arch is None else list(self.arch)
    new_arch += arch
    self.set_arch(new_arch)
    if len(arch) > 1 or arch_tuple:
      return self._subobjects[-len(arch):]
    else:
      return self._subobjects[-1]

#-------------------------------------------------------------------------------
  def update_arch(self): # invoked by level changes
    arch = []
    for subobject in self._subobjects:
      arch.append(subobject.arch)
    self.arch = list(arch)
    return self.arch

#-------------------------------------------------------------------------------
  def set_skipcv(self, scv = None, *scv_args, **scv_kwds):
    """
    scv is a vergence specification that unites outputs across levels. When
    referring to level outputs, this would be _post_ any output vergences 
    across streams within levels.

    For a stack with multioutput levels, scv must take the form of a list of tuples
    of lists, where:

    scv is a list with a length of the number of levels
    scv[i] is a tuple with a length of the number of outputs at the ith level, with
              None values for level outputs that undergo no skip vergence except for...
    scv[i][j] is a two-element list specifying: [skip_level_index, skip_output_index]

    The skip_level_index can be absolute or relative (e.g. -2 for two-levels prior).

    Although scv must be a list (or just None), skip vergence specifications for
    simpler stacks are optionally possible:

    If level[i] has a single output, scv[i] can be a list.
    If level[skip_level_index] has a single output, scv[i][j] can be an integer.
    If both the above apply, scv[i] can be an integer.
    Note Nones can be inserted at any point to flag omitted skip vergences.

    Concatenation ('con') is assumed but under `kwds', 'vergence_fn' may be 'sum'.

    The positition of the skip data can be specified in reference to the level
    input (skip_end = 'inp') or output (skip_end = 'out' - the default).

    More complicated connectivity patterns are possible by creating additional 
    streams across levels, with the possibility of using identity architectures 
    for individual streams to create effective skip vergences.
    """

    self.scv = scv
    self.scv_args = scv_args
    self.scv_kwds = dict(scv_kwds)
    if self.scv is None: return
    if 'vergence_fn' not in self.scv_kwds:
      self.scv_kwds.update({'vergence_fn': DEFAULT_VERGENCE_FUNCTION})
    if 'skip_end' not in self.scv_kwds:
      self.scv_kwds.update({'skip_end': DEFAULT_SKIP_END})
    if 'axis' not in self.scv_kwds:
      ax = 0
      if Creation('con') == Creation(self.scv_kwds['vergence_fn']):
        ax = -1
      self.scv_kwds.update({'axis': ax})
    # Quick check specification just for data type and length
    # (detailed skip connection convergence specification checks depend on 
    # levels vergences so this must wait until setup stage).
    if type(self.scv) is not list:
      raise TypeError("Skip vergence specification must be None or a list")
    elif len(self.scv) != self._n_subobjects:
      raise ValueError("Skip vergence specification list length must equal number of levels")


#-------------------------------------------------------------------------------
  def set_ipverge(self, spec = None, *args, **kwds):
    """
    spec = ipv is the vergence specification for stream inputs within levels.
    """
    return self._set_spec(self._subobject.set_ipverge, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_is_training(self, spec = None, *args, **kwds):
    """
    spec = is_training must be set to handle some operations (e.g. batch normalisation)
    """
    return self._set_spec(self._subobject.set_is_training, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_order(self, spec = None, *args, **kwds):
    """
    spec = 'datn' means order of: `dropout' `architecture', 'transfer function', 'normalisation'
    """
    return self._set_spec(self._subobject.set_order, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_biases(self, spec = None, *args, **kwds):
    """
    spec enables/disables biases and initialisation. Valid inputs are:
    None (default bias settings), False/True, disable/enable biases,
    or Bias initializer (e.g. 'zoi'): use bias with this initialiser
    """
    return self._set_spec(self._subobject.set_biases, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_weights(self, spec = None, *args, **kwds):
    """
    Sets initialiser for weights
    wgt = None or 'vs' (variance scaling)
    """
    return self._set_spec(self._subobject.set_weights, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_dropout(self, spec = None, *args, **kwds):
    """
    spec = None: No dropout
    spec = 0.: Full dropout (i.e. useless)
    spec = 0.4: dropout with keep probability of 0.6
    """
    return self._set_spec(self._subobject.set_dropout, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_transfn(self, spec = None, *args, **kwds):
    """
    spec = 'relu': ReLU
    spec = 'elu': ELU
    other options: 'softmax', and 'sigmoid'
    """
    argout = self._set_spec(self._subobject.set_transfn, spec, *args, **kwds)
    self.trans_fn = self[-1].trans_fn
    return argout

#-------------------------------------------------------------------------------
  def set_padwin(self, spec = None, *args, **kwds):
    """
    spec = 'same' or 'valid'
    """
    return self._set_spec(self._subobject.set_padwin, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_kernfn(self, spec = None, *args, **kwds):
    """
    spec = 'max' or 'avg'
    """
    return self._set_spec(self._subobject.set_kernfn, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_normal(self, spec = None, *args, **kwds):
    """
    spec = 'batch_norm' or 'lresp_norm' with accompanying keywords required.
    """
    return self._set_spec(self._subobject.set_normal, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_reguln(self, spec = None, *args, **kwds):
    """
    spec = 'l1_reg' or 'l2_reg' with accompanying keywords (scale) required.
    """
    return self._set_spec(self._subobject.set_reguln, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_opverge(self, spec = None, *args, **kwds):
    """
    spec = opc is the vergence specification for stream outputs within levels.
    """
    return self._set_spec(self._subobject.set_opverge, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = stack()
    elif not isinstance(other, stack) and not issubclass(other, stack):
      raise TypeError("Cannot clone to target class " + str(other))

    # Clone the architecture - this _will_ create new instances of streams
    if other._subobjects is None:
      other.set_arch(self.arch)
    elif self.arch != other.arch:
      raise AttributeError("Cannot clone to a target instance with differing architecture")

    # Clone the streams
    for self_subobject, other_subobject in zip(self._subobjects, other._subobjects):
      self_subobject.clone(other_subobject)

    # Now the rest of the stack-rank specifications
    if self.scv is not None: other.set_skipcv(self.scv, *self.scv_args, **self.scv_kwds)

    # Copy over the summary transfer function

    other.trans_fn = self.trans_fn

    # Rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)

    return other

#-------------------------------------------------------------------------------
  def __call__(self, inp = None, _called = True):
    # stack really doesn't care about nursemaiding inputs and outputs
    # because that's the job of levels. But we have to deal with skip
    # connection vergences.
    self._inp = inp
    if self._inp is None: return self._inp # nothing in, nothing out
    self.skip_verge = [None] * self._n_subobjects
    for i in range(self._n_subobjects):
      inp = self._subobjects[i].__call__(inp)
      inp = self._call_skipcv(inp, i)
    self.set_called(_called)
    self.arch_out = self[-1].arch_out
    self.pre_trans = self[-1].pre_trans
    self._out = inp
    self._setup_outputs() # concatenate output list of dictionaries
    return self.ret_out()

#-------------------------------------------------------------------------------
  def _call_skipcv(self, inp, index):
    """
    3 stages (in attempt to save headaches for the network designer): 

    1. Return if not relevant 
    2. Check specification with any errors trying to be informative 
    3. Setup specification
    """

    # Return if not relevant
    if self.scv is None or inp is None: return inp

    # Check the skip vergence specification 
    Inp, i = list(inp), int(index)
    n_inputs = len(Inp)
    unit_input = n_inputs == 1
    check_unit_skip = False

    if self.scv[i] is None:
      pass
    elif type(self.scv[i]) is not tuple: # Allow shorthand specifications for unit outputs
      if not(unit_input):
        raise ValueError("Non-tuple skip vergence specification possible only to single stream levels")
      elif type(self.scv[i]) is list or type(self.scv[i]) is int:
        self.scv[i] = (self.scv[i],)
      else:
        raise TypeError("Unrecognised skip vergence specification: " + str(self.scv[i]))
    elif len(self.scv[i]) != n_inputs:
      raise ValueError("Skip vergence specification incommensurate with number of output.")
    if self.scv[i] is None:
      pass
    else: # we need to complete any incomplete scv specifications
      scvi_list = list(self.scv[i])
      for j, scv_spec in enumerate(scvi_list): 
        if scv_spec is not None and type(scv_spec) is not list:
          if type(scv_spec) is int:      # Allow shorthand specifications for unit skip outputs
            scv_spec = [scv_spec, 0] 
            check_unit_skip = True       # - enable flag to check this requirement is fulfilled
          else:
            raise TypeError("Unrecognised skip vergence specification")
        if scv_spec is None:
          pass
        elif len(scv_spec) != 2:
          raise ValueError("Skip vergence specification target specification must be a list of two.")
        elif scv_spec[0] >= i:
          raise ValueError("Skip vergence target must precede connection level")
        elif scv_spec[0] < 0: # scvij = 0 is treated as an absolute reference
          scv_spec[0] += i
        # Now check unit-skip if required
        if check_unit_skip:
          if self.scv_kwds['skip_end'] == 'out':
            n_skips = len(self._subobjects[scv_spec[0]].ret_out())
          elif self.scv_kwds['skip_end'] == 'inp':
            n_skips = len(self._subobjects[scv_spec[0]].ret_inp())
          else:
            raise ValueError("Unknown skip end specification: " + str(self.scv_kwds['skip_end']))
          if n_skips != 1:
            raise ValueError("Integer specification for skip source requires a single output, not " + str(n_skips))
        scvi_list[j] = scv_spec
      self.scv[i] = tuple(scvi_list)

    # Now setup specification
    if self.scv[i] is not None:
      self.lastind = i
      self.skip_verge[i] = [None] * n_inputs
      for j in range(n_inputs):
        if self.scv[i][j] is not None:
          IJ = self.scv[i][j]
          if self.scv_kwds['skip_end'] == 'out':
            skip_inputs = list(self._subobjects[IJ[0]].ret_out())
          elif self.scv_kwds['skip_end'] == 'inp':
            skip_inputs = list(self._subobjects[IJ[0]].ret_inp())
          else:
            raise ValueError("Unknown skip end specification: " + str(self.scv_kwds['skip_end']))
          inputs = [Inp[j], skip_inputs[IJ[1]]]
          func = self.scv_kwds['vergence_fn']
          kwds = dict(self.scv_kwds)
          kwds.pop('vergence_fn')
          kwds.pop('skip_end')
          skip_name = self.name + "/" + self._subobject_name + "s_"
          skip_name += str(i) + "_and_" + str(IJ[0]) + "_skip/" 
          skip_name += func + "vergence_" + str(j)
          self.skip_verge[i][j] = Creation(func)(inputs, *self.scv_args, 
                                  name = skip_name, **kwds)
          Inp[j] = self.skip_verge[i][j]
    return tuple(Inp)

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
