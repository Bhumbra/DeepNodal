# Leaf abstract class for Tensorflow. A leaf is the lowest hierarchical structure
# that supports parameters. It is abstract and the instantiable inheriting class
# is link.
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.structure import *
from deepnodal.python.concepts.mapping import *

#-------------------------------------------------------------------------------

class leaf (structure):
  """
  A leaf is a structure that possess no hierarchical substructures and is the
  initiator of parameters where self.params comprise a list of ordered 
  dictionaries.

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
        if not(isinstance(param_dict, mapping)):
          raise TypeError("Only mappings are accepted parameters.")
    return self.ret_params()

#-------------------------------------------------------------------------------
  def add_param(self, param_dict):
    if not isinstance(param_dict, mapping):
      raise TypeError("Only mappings are accepted parameters.")
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

def ret_unique(sources, spec = None):
  """
  Returns a single value ordered dictionary {name:object} according to argument 
  sources. Argument sources may take the following forms:

  - if type(sources) == Creation('var'): returned (name inferred from object).

  - if type(sources) is tuple (minimum length 2) in the following form:
    (structure_instance, callable_member_function, arg3, arg4, ...)

  - if type(sources) is list (i.e. in the form of stem._params):
    - if len(spec) is 1 and contains a single parameter dictionary.
    - otherwise the list is filtered by name.
  """
  type_sources = type(sources)

  # Variable case
  if type_sources is not list and type_sources is not tuple: 
    value, value_name = sources, sources.name
    if spec is not None:
      if not(spec in value_name):
        raise ValueError("Value name " + value_spec + " not found in " + str(value))
    return {value_name: value}

  # Tuple case
  if type_sources is tuple:
    if len(sources) < 2:
      raise ValueError("Minimum sources tuple length of 2")
    else:
      if not(isinstance(sources[0], structure)):
        raise TypeError("First element of sources tuple must be a structure instance")
      if not(callable(sources[1])):
        raise TypeError("Second element of sources tuple must be callable")
      sources = sources[1](sources[0], *sources[2:])
      type_sources = type(sources)
 
  # List case
  if type_sources is not list:
    raise TypeError("Unrecognised value type specification: " + str(type_sources))
  values = []
  for source in sources:
    type_source = type(source)
    if type_source is not dict:
      raise TypeError("Unrecognised value type: " + str(type_source))
    if len(source) != 1:
      raise ValueError("Parameter dictionary must be of size one")
    source_name = list(source)[0]
    append_value = True if spec is None else spec in source_name
    if append_value:
      values.append(source)

  # Test for uniqueness
  n_values = len(values)
  if n_values != 1:
    raise ValueError("No unique value with " + str(n_values) + " candidates")

  return values[0]

#-------------------------------------------------------------------------------
def structuref2unique(*_args, **_kwds):
  """
  This function converts structure references within args and kwds to a unique 
  dictionaries whether a parameter or output.
  """
  args = list(_args)
  kwargs = list(_kwds)
  kwvals = [_kwds[kwarg] for kwarg in kwargs]
  for i, arg in enumerate(args):
    if type(arg) is tuple:
      if len(arg) > 1:
        if isinstance(arg[0], structure):
          if callable(arg[1]):
            args[i] = ret_unique(arg)
  kwds = dict()
  for kwarg, kwval in zip(kwargs, kwvals):
    arg, val = kwarg, kwval
    if type(val) is tuple:
      if len(val) > 1:
        if isinstance(val[0], structure):
          if callable(val[1]):
            val = ret_unique(val)
    kwds.update({arg:val})
  return tuple(args), dict(kwds)

#-------------------------------------------------------------------------------

