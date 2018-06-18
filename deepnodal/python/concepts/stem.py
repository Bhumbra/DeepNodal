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

  The stem class is abstract and inheriting classes must define self.__call__(inp)
  to be instantiated.
  """

  # public
  def_name = 'stem'                 # default name
  def_subobject = leaf              # default subobject class
  def_subobject_name = 'subobject'  # default subobject name

  # protected
  _subobjects = None                # subobject instances which may be leaves or stems
  _n_subobjects = None              # number of subobjects
  _subobject = leaf                 # subobject class
  _unit_subobject = None            # flag to state unit subobject
  _subobject_name = 'subobject'     # default subobject name if not unit_subobject
  _spec_type = None                 # container type to give specifications to subobjects (dict not allowed)
  _inp = None                       # input
  _out = None                       # output
  _params = None                    # parameters - collated from subobjects
  _n_params = None                  # len(parameters)
  _outputs = None                   # parameters - collated from subobjects
  _n_outputs = None                 # len(parameters)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_subobject()
    self.set_subobjects()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self._subobjects is None: return
    for i, subobject in enumerate(self._subobjects):
    # no point re-naming subobjects if unit_subobject is true 
      subobject_name = self.name if self._unit_subobject else self.name + "/" + self._subobject_name + "_" + str(i)
      subobject.set_name(subobject_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev
    if self._subobjects is None: return
    for subobject in self._subobjects:
      subobject.set_dev(dev)

#-------------------------------------------------------------------------------
  def set_subobject(self, subobject = None, subobject_name = None):
    """
    This sets the class-type of each subobject and associated name prior to indexing.
    """
    self._subobject = subobject if subobject is not None else self.def_subobject
    self._subobject_name = subobject_name if subobject_name is not None else self.def_subobject_name

#-------------------------------------------------------------------------------
  def set_subobjects(self, subobjects = None):
    """
    This allows either manually setting the list of subobjects, or if subobjects is an
    integer it instantiates that number of subobjects.
    """
    self._subobjects = subobjects
    self._n_subobjects = 0
    self._unit_subobject = None
    if self._subobjects is None:
      return self._subobjects
    elif type(self._subobjects) is list:
      self._n_subobjects = len(self._subobjects)
      self._unit_subobject = self._n_subobjects == 1
      # it would be quite rude to rename or redevice these subobjects so we won't
    elif type(subobjects) is int:
      self._n_subobjects = subobjects
      self._unit_subobject = self._n_subobjects == 1
      self._subobjects = [self._subobject() for i in range(self._n_subobjects)]
      self.set_name(self.name) # this renames all subobjects
      self.set_dev(self.dev)   # this redevices all subobjects
    else:
      raise TypeError("Unrecognised subobjects specification.")
    return self._subobjects

#-------------------------------------------------------------------------------
  def set_inp(self, inp = None):
    self._inp = inp
    self._out = None
    return self.inp
#-------------------------------------------------------------------------------
  def __getitem__(self, index):
    return self._subobjects[index]

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._n_subobjects

#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, inp = None): # this function is for calling graph objects
    pass

#-------------------------------------------------------------------------------
  def ret_inp(self):
    return self._inp

#-------------------------------------------------------------------------------
  def ret_out(self):
    return self._out

#-------------------------------------------------------------------------------
  def _set_spec(self, func, spec = None, *args, **kwds):
    """
    Allows set_specing of a specification to all subobjects in the following form:
    return [func(subobject, spec, *args, **kwds) for subobject in enumerate(self.subobjects)]
    ...or...
    return [func(subobject, spec[i], *args[i], **kwds[i]) for i, subobject in enumerate(self.subobjects)]
    ...or...
    any such combination.
    """
    if self._spec_type is None:
      return [func(subobject, spec, *args, **kwds) for i, subobject in enumerate(self._subobjects)]
    if type(spec) is not self._spec_type:
      spec = self._spec_type([spec] * self._n_subobjects)
    elif len(spec) != self._n_subobjects:
      raise ValueError("Specification incommensurate with number of subobjects")
    if len(kwds):
      if len(args) == 1:
        args = args[0]
        if type(args) is not self._spec_type:
          args = self._spec_type([args] * self._n_subobjects)
        elif len(args) != self._n_subobjects:
          raise ValueError("Specification arguments incommensurate with number of subobjects")
        return [func(subobject, spec[i], args[i], **kwds) for i, subobject in enumerate(self._subobjects)]
      elif len(args):
        return [func(subobject, spec[i], *args, **kwds) for i, subobject in enumerate(self._subobjects)]
      else:
        return [func(subobject, spec[i], **kwds) for i, subobject in enumerate(self._subobjects)]
    elif len(args) == 1:
      args = args[0]
      if type(args) is dict:
        kwds = dict(args)
        return [func(subobject, spec[i], **kwds) for i, subobject in enumerate(self._subobjects)]
      elif type(args) is self._spec_type:
        if len(args) != self._n_subobjects:
          raise ValueError("Specification arguments incommensurate with number of subobjects")
        return [func(subobject, spec[i], args[i]) for i, subobject in enumerate(self._subobjects)]
      else:
        return [func(subobject, spec[i], args) for i, subobject in enumerate(self._subobjects)]
    elif len(args):
      return [func(subobject, spec[i], args) for i, subobject in enumerate(self._subobjects)]
    else:
      return [func(subobject, spec[i]) for i, subobject in enumerate(self._subobjects)]

#-------------------------------------------------------------------------------
  def _setup_params(self):
    """
    Collates lists of parameter dictionaries to a single list self.params.
    Classes inheriting from stemes do not possess autonomous parameter lists
    but must collate their lists from subobjects, until eventually reaching leaf-derived
    classes each of which may posses an autonomous parameter list associated with
    a single TensorFlow call.
    """
    self._params = []
    for subobject in self._subobjects:
      subobject._setup_params()
      self._params += subobject._params
    self._n_params = len(self._params)
    return self._params

#-------------------------------------------------------------------------------
  def ret_params(self, param_spec = None, ret_indices = False):
    if self._params is None:
      self._setup_params()
    elif not(len(self._params)):
      self._setup_params()
    if param_spec is None:
      if not ret_indices:
        return self._params
      else:
        return list(range(len(self._params)))

    params = []
    indices = []
    if type(param_spec) is bool:
      if param_spec:
        params = self._params
        indices = list(range(len(self._params)))
    elif len(param_spec) != self._n_subobjects:
      raise ValueError("Parameter specification incommensurate with hierarchical structure")
    else:
      for i, spec in enumerate(param_spec):
        params += self._subobjects[i].ret_params(spec)
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
  def _setup_outputs(self):
    """
    Collates lists of output dictionaries to a single list self.outputs.
    Classes inheriting from stemes do not possess autonomous outputs lists
    but must collate their lists from subobjects, until eventually reaching leaf-derived
    classes each of which may posses an autonomous output list associated with
    a single TensorFlow call.
    """
    self._outputs = []
    for subobject in self._subobjects:
      subobject._setup_outputs()
      self._outputs += subobject._outputs
    self._n_outputs = len(self._outputs)
    return self.ret_outputs()

#-------------------------------------------------------------------------------
  def ret_outputs(self, outputs_spec = None):
    if outputs_spec is None:
      return self._outputs

    outputs = []
    if type(outputs_spec) is bool:
      if outputs_spec:
        outputs = self._outputs
    elif len(outputs_spec) != self._n_subobjects:
      raise ValueError("Outputs specification incommensurate with hierarchical structure")
    else:
      for i, spec in enumerate(outputs_spec):
        outputs += self._subobjects[i].ret_outputs(spec)
    return outputs

#-------------------------------------------------------------------------------

