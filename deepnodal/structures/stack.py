"""
Stack module for Tensorflow. A stack is a structure that contains one or more
levels in series with additional support for uninterrupted skip connections.
If skip-connections with intervening computations are required, then this can
be created using multiple streams across levels using (None, arch_spec) arch
specifications where 'None' denotes only an `identity'.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.concepts.stem import stem
from deepnodal.structures.level import *

#-------------------------------------------------------------------------------
class stack (stem):
  # A stack comprises a single level or multiple levels. Stacks can be connected
  # to one or more stacks at both input and output ends. Therefore in principle,
  # multiple stacks should be all that is necessary to design any network.

  def_name = 'stack'
  def_subobject_name = 'level'
  def_subobject = level
  arch = None             # architecture
  type_arch = None        # level architectures if all stream types are the same
  arch_out = None         # archecture output of last level if unit_stream
  skc = None              # Skip coalescence specification (across levels)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(name, dev)
    self.set_arch() # defaults to an identity
    self.setup():

#-------------------------------------------------------------------------------
  def set_arch(self, arch = None):
    self.arch = arch
    if type(self.arch) is not list:
      if self.arch is None or type(self.arch) is tuple:
        self.arch = [self.arch]
      elif type(self.arch) is int: # ! sloppy but barely acceptable
        self.arch = [(self.arch,)]
      else:
        raise TypeError("Unknown level architecture: " + str(self.arch))
    self.set_subobjects(len(self.arch))
    self.type_arch = [None] * self.n_subobjects
    for i, arch in enumerate(self.arch):
      self.subobjects[i].set_arch(arch)
      self.type_arch[i] = self.subobjects[i].type_arch
    return self.type_arch

#-------------------------------------------------------------------------------
  def set_skip_coal(self, skc = None, *skc_args, **skc_kwds):
    """
    skc is a coalescence specification that unites stream outputs across levels.

    For stack with multistream levels, skc must take the form of a list of tuples
    of lists, where:

    skc is a list with a length of the number of levels
    skc[i] is a tuple with a length of the number of streams at the ith level, with
              None values for streams that undergo no skip coalescence except for...
    skc[i][j] is a two-element list specifying: [skip_level_index, skip_stream_index]

    The skip_level_index can be absolute or relative (e.g. -2 for two-levels prior).

    Although skc must be a list (or just None), skip coalescence specifications for
    simpler stacks are optionally possible:

    If level[i] has a single stream, skc[i] can be a list.
    If level[skip_level_index] has a single stream, skc[i][j] can be an integer.
    If both the above apply, skc[i] can be an integer.
    Note Nones can be inserted at any point to flag omitted skip coalescences.

    Concatenation ('con') is assumed but under `kwds', 'coalescence_fn' may be 'sum'.

    More complicated connectivity patterns require creating additional streams across
    levels, with the possibility of using identity architectures for individual
    streams to create effective skip coalescences.

    """
    self.skc = skc
    self.skc_args = skc_args
    self.skc_kwds = dict(skc_kwds)
    if self.skc is None: return
    if type(self.skc) is not list:
      raise TypeError("Skip coalescence specification must be None or a list")
    elif len(self.skc) != self.n_subobjects:
      raise ValueError("Skip coalescence specification list length must equal number of levels")
    for i in range(self.n_subobjects):
      if self.sck[i] is None:
        pass
      elif type(self.skc[i] is not tuple):
        if not(self.subobjects[i].unit_subobject):
          raise ValueError("Non-tuple skip coalescence specification possible only to single stream levels")
        elif type(self.skc[i]) is list or type(self.skc[i]) is int:
          self.skc[i] = (self.skc[i],)
        else:
         raise TypeError("Unrecognised skip coalescence specification")
      elif len(self.skc[i] != self.levels[i].n_subobjects):
        raise ValueError("Skip coalescence specification incommensurate with number of streams at output level.")
      skci = list(self.skci)
      for j in range(self.subobjects[i].n_subobjects):
        skcij = skci[j]
        if skci[j] is None:
          pass
        elif type(skcij) is int:
          if skcij == 0:
            skij = None
          elif skcij >= i:
            raise ValueError("Skip coalescence target must precede connection level")
          elif skcij < 0:
            skcij += i
          if not(self.subobjects[sckcij].unit_subobject):
            raise ValueError("Skip coalescence specification incommensurate with number of streams at skip level.")
          elif skcij is not None:
            skci[j] = [skcij, 0]
        elif type(skcij) is list:
          if len(skcij) != 2:
            raise ValueError("Skip coalescence specification target specification must be a list of two.")
          elif skcij[0] == 0:
            skcij = None
          elif skcij[0] >= i:
            raise ValueError("Skip coalescence target must precede connection level")
          elif skcij[0] < 0:
            skcij[0] += i
          if skcij is not None:
            if skcij[1] >= self.subobjects[sckij[0]].n_subobjects:
              raise ValueError("Skip coalescence specification exceeds target specification stream number.")
        skci[j] = skcij
      self.skc[i] = tuple(skci)
    if not(len(self.skc_kwds)):
      self.skc_kwds = self.skc_kwds.update({'coalescence_fn': 'con'})
    if 'axis' not in self.skc_kwds:
      self.skc_kwds = self.skc_kwds.update({'axis': -1})

#-------------------------------------------------------------------------------
  def set_input_coal(self, spec = None, *args, **kwds):
    """
    spec = ipc is the coalescence specification for stream inputs within levels.
    """
    return self.broadcast(self.subobject.set_input_coal, spec, *args, **kwds)

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
  def set_reguln(self, spec = None, *args, **kwds):
    """
    spec = 'l1_reg' or 'l2_reg', with keyword: scale=scale
    """
    return self.broadcast(self.subobject.set_reguln, spec, *args, **kwds)

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
    return self.broadcast(self.subobject.set_transfn, spec, *args, **kwds)

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
  def set_output_coal(self, spec = None, *args, **kwds):
    """
    spec = opc is the coalescence specification for stream outputs within levels.
    """
    return self.broadcast(self.subobject.set_input_coal, spec, *args, **kwds)

#-------------------------------------------------------------------------------
  def setup(inp = None):
    # stack really doesn't care about nursemaiding inputs and outputs
    # because that's the job of levels. But we have to deal with skip
    # coalescences.
    self.inp = inp
    self.skip_coal = [None] * self.n_subobjects
    for i in range(self.n_subobjects):
      inp = self.subobjects[i].setup(inp)
      if self.skc is not None:
        Inp = list(inp)
        if self.skc[i] is not None:
          self.skip_coal[i] = [None] * self.subobjects[i].n_subobjects
          for j in range(self.subobjects[i].n_subobjects):
            if self.skc[i][j] is not None:
              func = self.skc_kwds['coalescence_fn']
              kwds = self.kwc_kwds.pop('coalescence_fn')
              self.skip_coal[i][j] = Creation(func)(inp[j], *self.skc_args, **kwds
                                     name = self.name + "/skip_" + func + "_" + str(i))
              Inp[j] = self.skip_coal[i][j]
        inp = tuple(Inp)
    self.arch_out = self.subobjects[-1].arch_out
    self.out = inp

#-------------------------------------------------------------------------------

