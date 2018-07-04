# Leaf abstract class for Tensorflow. A leaf is the lowest hierarchical structure
# that supports parameters. It is abstract and the instantiable inheriting class
# is link.
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.structure import *

#-------------------------------------------------------------------------------

class leaf (structure):
  """
  A leaf is a structure that possess no hierarchical substructures and is the
  initiator of parameters where self.params comprise a list of dictionaries.

  A leaf is an abstract class and requires an inheriting class with an explicit
  self.__call__(inp) function for instantiation.
  """
  # public
  def_name = 'leaf'   # default name

  # protected
  _inp = None         # input
  _out = None         # output
  _params = None      # a list of dictionaries containing parameter objects.
  _n_params = None    # number of parameters

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.setup_params()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, inp = None): # this function is for calling graph objects
    pass

#-------------------------------------------------------------------------------
  def set_inp(self, inp = None):
    self._inp = inp
    self._out = None
    return self._inp

#-------------------------------------------------------------------------------
  def ret_inp(self):
    return self._inp

#-------------------------------------------------------------------------------
  def ret_out(self):
    return self._out

#-------------------------------------------------------------------------------
  def setup_params(self, params = None):
    self._params = params
    if self._params is None: self._params = []
    self._n_params = len(self._params)
    if self._n_params:
      for param_dict in self._params:
        if type(param_dict) is not dict:
          raise TypeError("Only dictionaries are accepted parameters.")
    return self.ret_params()

#-------------------------------------------------------------------------------
  def add_param(self, param_dict):
    if type(param_dict) is not dict:
      raise TypeError("Only dictionaries are accepted parameters.")
    self._params.append(param_dict)
    self._n_params = len(self._params)
    return self.ret_params()

#-------------------------------------------------------------------------------
  def ret_params(self):
    if not(self._called): return self, leaf.ret_params
    if self._params is None: self.set_params()
    return self._params

#-------------------------------------------------------------------------------

# Leaf-associated functions
#-------------------------------------------------------------------------------

def ret_unique_param(self, sources, param_spec = None):
  """
  Returns a single parameter (object, name) tuple according to argument sources. 
  Argument sources may take the following forms:

  #1 if type(sources) == Creation('var'): returned (name inferred by object name).

  #2 if isinstance(sources, structure): sources.ret_params(param_spec)

  #3 if type(sources) is tuple:
    - if len(sources) == 1 and isinstance(sources[0], structure) see #2
    - otherwise if sources[1] is callable: sources[1](sources[0], *sources[2:])

  #4 if type(sources) is list (i.e. in the form of stem._params):
    - if len(spec) is 1 and contains a single parameter dictionary.
    - otherwise the parameter list is filtered by name (param_spec).
  """
  type_sources = type(sources)
  check_param_spec = param_spec is not None

  # Variable case
  if type_sources == Creation('var'): 
    param, param_name = sources, sources.name
    if check_param_spec:
      if not(param_spec in param_name):
        raise ValueError("Parameter name " + param_spec + " not found in " + str(param))
    return param, param_name

  # Raise error for unrecognised sources types
  if type_sources is not list and type_sources is not tuple and not(isinstance(sources, structure)):
    raise TypeError("Unrecognised sources type: " + str(type_sources))

  # Structure case - deal with single tuple case first
  if type_sources is tuple and len(sources) == 1:
    sources = sources[0]
    type_sources = type(sources)
    if not isinstance(sources, structure):
      raise TypeError("Unrecognised single tuple format: " + str(type_sources))

  # Structure case
  if isinstance(sources, structure):
    sources = sources.ret_params(param_spec)
    check_param_spec = False
    type_sources = type(sources)
    if type_sources is tuple:
      raise ValueError("Cannot return parameter before parameter object is called")

  # Tuple case
  if type_sources is tuple:
    if len(sources) < 1:
      raise ValueError("Minimum sources tuple length of 1")
    else: # by this stage len(sources) > 1
      if not(callable(sources[1])):
        raise TypeError("Second element of sources when a multiple tuple must be callable")
      sources = sources[1](sources[0], *sources[2:])
      type_sources = type(sources)
 
  # List case
  if type_sources is not list:
    raise TypeError("Unrecognised parameter specification: " + str(type_sources))

  param, param_name = [], []
  for source in sources:
    type_source = type(source)
    if type_source is not dict:
      raise TypeError("Unrecognised parameter type: " + str(type_source))
    if len(source) != 1:
      raise ValueError("Parameter dictionary must be of size one")
    source_name = list(source)[0]
    append_param = True if not check_param_spec else param_spec in source_name
    if append_param:
      param.append(source[source_name])
      param_name.append(source_name)

  # Test for uniqueness
  n_param = len(param)
  if n_param != 1:
    raise ValueError("No unique parameter with " + str(n_param) + " candidates")

  return param[0], param_name[0]

#-------------------------------------------------------------------------------

