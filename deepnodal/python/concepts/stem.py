# Stem module for Tensorflow. A stem is a structure that contains one or more
# substructures called subobjects. These subobjects may be classes that inherit from stem
# or from leaf. Note it is up to inheriting classes whether the subobjects are arranged
# in series, in parallel, or a combination.
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.leaf import *

#-------------------------------------------------------------------------------
class stem (structure): # we inherit structure because a stem is never a leaf
  """
  A stem is a structure that supports and broadcasts specifications to many subobjects.
  Note a subobject may be another stem or a leaf (the default).

  The stem class is abstract and inheriting classes must define self.setup(inp)
  to be instantiated.
  """

  def_name = 'stem'                # default name
  def_subobject = leaf             # default subobject class
  def_subobject_name = 'subobject' # default subobject name
  subobjects = None                # subobject instances which may be leaves or stems
  n_subobjects = None              # number of subobjects
  subobject = leaf                 # subobject class
  unit_subobject = None            # flag to state unit subobject
  subobject_name = 'subobject'     # default subobject name if not unit_subobject
  spec_type = None                 # container type to give specifications to subobjects (dict not allowed)
  inp = None                       # input
  out = None                       # output
  params = None                    # parameters - collated from subobjects
  n_params = None                  # len(parameters)
  outputs = None                   # parameters - collated from subobjects
  n_outputs = None                 # len(parameters)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_subobject()
    self.set_subobjects()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self.subobjects is None: return
    for i, subobject in enumerate(self.subobjects):
    # no point re-naming subobjects if unit_subobject is true 
      subobject_name = self.name if self.unit_subobject else self.name + "/" + self.subobject_name + "_" + str(i)
      subobject.set_name(subobject_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev
    if self.subobjects is None: return
    for subobject in self.subobjects:
      subobject.set_dev(dev)

#-------------------------------------------------------------------------------
  def set_subobject(self, subobject = None, subobject_name = None):
    """
    This sets the class-type of each subobject and associated name prior to indexing.
    """
    self.subobject = subobject if subobject is not None else self.def_subobject
    self.subobject_name = subobject_name if subobject_name is not None else self.def_subobject_name

#-------------------------------------------------------------------------------
  def set_subobjects(self, subobjects = None):
    """
    This allows either manually setting the list of subobjects, or if subobjects is an
    integer it instantiates that number of subobjects.
    """
    self.subobjects = subobjects
    self.n_subobjects = 0
    self.unit_subobject = None
    if self.subobjects is None:
      return self.subobjects
    elif type(self.subobjects) is list:
      self.n_subobjects = len(self.subobjects)
      self.unit_subobject = self.n_subobjects == 1
      # it would be quite rude to rename or redevice these subobjects so we won't
    elif type(subobjects) is int:
      self.n_subobjects = subobjects
      self.unit_subobject = self.n_subobjects == 1
      self.subobjects = [self.subobject() for i in range(self.n_subobjects)]
      self.set_name(self.name) # this renames all subobjects
      self.set_dev(self.dev)   # this redevices all subobjects
    else:
      raise TypeError("Unrecognised subobjects specification.")
    return self.subobjects

#-------------------------------------------------------------------------------
  def set_inp(self, inp = None):
    self.inp = inp
    self.out = None
    return self.inp

#-------------------------------------------------------------------------------
  @abstractmethod
  def setup(self, inp = None): # this function is for creating graph objects
    pass

#-------------------------------------------------------------------------------
  def ret_inp(self):
    return self.inp

#-------------------------------------------------------------------------------
  def ret_out(self):
    return self.out

#-------------------------------------------------------------------------------
  def set_spec(self, func, spec = None, *args, **kwds):
    """
    Allows set_specing of a specification to all subobjects in the following form:
    return [func(subobject, spec, *args, **kwds) for subobject in enumerate(self.subobjects)]
    ...or...
    return [func(subobject, spec[i], *args[i], **kwds[i]) for i, subobject in enumerate(self.subobjects)]
    ...or...
    any such combination.
    """
    if self.spec_type is None:
      return [func(subobject, spec, *args, **kwds) for i, subobject in enumerate(self.subobjects)]
    if type(spec) is not self.spec_type:
      spec = self.spec_type([spec] * self.n_subobjects)
    elif len(spec) != self.n_subobjects:
      raise ValueError("Specification incommensurate with number of subobjects")
    if len(kwds):
      if len(args) == 1:
        args = args[0]
        if type(args) is not self.spec_type:
          args = self.spec_type([args] * self.n_subobjects)
        elif len(args) != self.n_subobjects:
          raise ValueError("Specification arguments incommensurate with number of subobjects")
        return [func(subobject, spec[i], args[i], **kwds) for i, subobject in enumerate(self.subobjects)]
      elif not(len(args)):
        return [func(subobject, spec[i], **kwds) for i, subobject in enumerate(self.subobjects)]
      else:
        return [func(subobject, spec[i], *args, **kwds) for i, subobject in enumerate(self.subobjects)]
    elif len(args) == 1:
      args = args[0]
      if type(args) is dict:
        kwds = dict(args)
        return [func(subobject, spec[i], **kwds) for i, subobject in enumerate(self.subobjects)]
      elif type(args) is self.spec_type:
        if len(args) != self.n_subobjects:
          raise ValueError("Specification arguments incommensurate with number of subobjects")
        return [func(subobject, spec[i], args[i]) for i, subobject in enumerate(self.subobjects)]
      else:
        return [func(subobject, spec[i], args) for i, subobject in enumerate(self.subobjects)]
    elif not(len(args)):
      return [func(subobject, spec[i]) for i, subobject in enumerate(self.subobjects)]
    else:
      return [func(subobject, spec[i], args) for i, subobject in enumerate(self.subobjects)]

#-------------------------------------------------------------------------------
  def setup_params(self):
    """
    Collates lists of parameter dictionaries to a single list self.params.
    Classes inheriting from stemes do not possess autonomous parameter lists
    but must collate their lists from subobjects, until eventually reaching leaf-derived
    classes each of which may posses an autonomous parameter list associated with
    a single TensorFlow call.
    """
    self.params = []
    for subobject in self.subobjects:
      subobject.setup_params()
      self.params += subobject.params
    self.n_params = len(self.params)
    return self.params

#-------------------------------------------------------------------------------
  def ret_params(self, param_spec = None, ret_indices = False):
    if param_spec is None:
      if not ret_indices:
        return self.params
      else:
        return list(range(len(self.params)))

    params = []
    indices = []
    if type(param_spec) is bool:
      if param_spec:
        params = self.params
        indices = list(range(len(self.params)))
    elif len(param_spec) != self.n_subobjects:
      raise ValueError("Parameter specification incommensurate with hierarchical structure")
    else:
      for i, spec in enumerate(param_spec):
        params += self.subobjects[i].ret_params(spec)
      for param in params:
        param_name = list(param)[0]
        param_object = param[param_name]
        for i, _param in enumerate(self.params):
          _param_name = list(_param)[0]
          if _param[_param_name] == param_object:
            indices.append(i)
    if not ret_indices:
      return params
    else:
      return indices

#-------------------------------------------------------------------------------
  def setup_outputs(self):
    """
    Collates lists of output dictionaries to a single list self.outputs.
    Classes inheriting from stemes do not possess autonomous outputs lists
    but must collate their lists from subobjects, until eventually reaching leaf-derived
    classes each of which may posses an autonomous output list associated with
    a single TensorFlow call.
    """
    self.outputs = []
    for subobject in self.subobjects:
      subobject.setup_outputs()
      self.outputs += subobject.outputs
    self.n_outputs = len(self.outputs)
    return self.ret_outputs()

#-------------------------------------------------------------------------------
  def ret_outputs(self, outputs_spec = None):
    if outputs_spec is None:
      return self.outputs

    outputs = []
    if type(outputs_spec) is bool:
      if outputs_spec:
        outputs = self.outputs
    elif len(outputs_spec) != self.n_subobjects:
      raise ValueError("Outputs specification incommensurate with hierarchical structure")
    else:
      for i, spec in enumerate(outputs_spec):
        outputs += self.subobjects[i].ret_outputs(spec)
    return outputs

#-------------------------------------------------------------------------------
