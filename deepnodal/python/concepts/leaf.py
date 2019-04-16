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
  _moments = None      # a list of dictionaries containing moment objects.
  _n_moments = None    # number of moments

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.setup_params()
    self.setup_moments()

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
    return self._params

#-------------------------------------------------------------------------------
  def add_param(self, param_dict):
    if not isinstance(param_dict, mapping):
      raise TypeError("Only mappings are accepted parameters.")
    self._params.append(param_dict)
    self._n_params = len(self._params)
    return self._params

#-------------------------------------------------------------------------------
  def ret_params(self, param_spec = None, ret_indices = False):
    """
    Returns parameter mappings (and associated indices if ret_indices is True)
    depending on the value of param_spec:

    param_spec = None (or True): returns all parameters
    param_spec = a string: returns parameters which includes that string in the name

    """
    if not(self._called): return self, leaf.ret_params, param_spec, ret_indices
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
    elif type(param_spec) is str:
      for i, param in enumerate(self._params):
        if param_spec in list(param)[0]:
          params.append(param)
          indices.append(i)
    if not ret_indices:
      return params
    else:
      return indices

#-------------------------------------------------------------------------------
  def _ret_param(self, param_spec = None):
    """
    Identical to self.ret_params but with checking of a unique result and returns
    the mapping itself.

    """
    if not(self._called): return self, leaf._ret_param, param_spec
    param = self.ret_params(param_spec)
    if len(param) != 1:
      raise ValueError("Specification " + str(param_spec) + 
                       " returns " + str(len(param)) + " results.")

    return param[0]

#-------------------------------------------------------------------------------
  def ret_param(self, param_spec = None):
    """
    Identical to self._ret_param but returns the graph object rather than mapping. 
    """
    if not(self._called): return self, leaf.ret_param, param_spec
    param = self._ret_param(param_spec)
    return list(param.values())[0]

#-------------------------------------------------------------------------------
  def setup_moments(self, moments = None):
    self._moments = moments
    if self._moments is None: self._moments = []
    self._n_moments = len(self._moments)
    if self._n_moments:
      for moment_dict in self._moments:
        assert isinstance(moment_dict, mapping),\
          "Only mappings are accepted moments"
    return self._params

#-------------------------------------------------------------------------------
  def add_moment(self, moment_dict):
    assert isinstance(moment_dict, mapping),\
      "Only mappings are accepted moments."
    self._moments.append(moment_dict)
    self._n_moments = len(self._moments)
    return self._moments

#-------------------------------------------------------------------------------
  def ret_moments(self):
    return self._moments

#-------------------------------------------------------------------------------

# Leaf-associated functions
#-------------------------------------------------------------------------------

def ret_unique(sources, spec = None):
  """
  Returns a single value mapping {name:object} according to argument 
  sources. Argument sources may take the following sequence:

  - if type(sources) is tuple (minimum length 2) in the following form:
    (structure_instance, callable_member_function, arg3, arg4, ...)

  - if isinstance(sources, mapping): returns ret_unique([sources])

  - if type(sources) == Creation('var'): returned (name inferred from object).

  - if type(sources) is list (e.g. in the form of stem._params):
    - if len(spec) is 1 and contains a single value mapping.
    - otherwise the list is filtered by name.
  """
  type_sources = type(sources)

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
 
  # Mapping case - convert to list
  if isinstance(sources, OrderedDict) or isinstance(sources, mapping):
    sources = [sources]
    type_sources = type(sources)

  # Variable case
  if type_sources is not list and type_sources is not tuple: 
    value, value_name = sources, sources.name.replace(':', '_')
    if spec is not None:
      if not(spec in value_name):
        raise ValueError("Value name " + value_spec + " not found in " + str(value))
    sources = [mapping({value_name: value})]
    type_sources = type(sources)
  
  # List case
  if type_sources is not list:
    raise TypeError("Unrecognised value type specification: " + str(type_sources))
  values = []
  for source in sources:
    type_source = type(source)
    if not isinstance(source, mapping):
      raise TypeError("Unrecognised value type: " + str(type(source)))
    if len(source) != 1:
      raise ValueError("Value dictionary must be of size one")
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

