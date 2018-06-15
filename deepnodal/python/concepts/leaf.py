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
  self.setup function for instantiation.
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
    return self._params

#-------------------------------------------------------------------------------

