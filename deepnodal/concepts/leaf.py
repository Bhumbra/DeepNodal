# Leaf abstract class for Tensorflow. A leaf is the lowest hierarchical structure
# that supports parameters. It is abstract and the instantiable inheriting class
# is link.
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.concepts.structure import *

#-------------------------------------------------------------------------------

class leaf (structure):
  """
  A leaf is a structure that possess no hierarchical substructures and is the
  initiator of parameters where self.params comprise a list of dictionaries.

  A leaf is an abstract class and requires an inheriting class with an explicit
  self.setup function for instantiation.
  """
  inp = None         # input
  out = None         # output
  def_name = 'leaf'  # default name
  params = None      # a list of dictionaries containing parameter objects.
  n_params = None    # number of parameters

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_params()
    self.set_outputs()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  @abstractmethod
  def setup(self, inp = None): # this function is for creating graph objects
    pass

#-------------------------------------------------------------------------------
  def set_inp(self, inp = None):
    self.inp = inp
    self.out = None
    return self.inp

#-------------------------------------------------------------------------------
  def ret_out(self, out = None):
    return self.out

#-------------------------------------------------------------------------------
  def set_params(self, params = None):
    self.params = params
    if self.params is None: self.params = []
    self.n_params = len(self.params)
    if self.n_params:
      for param_dict in self.params:
        if type(param_dict) is not dict:
          raise TypeError("Only dictionaries are accepted parameters.")
    return ret_params()

#-------------------------------------------------------------------------------
  def ret_params(self):
    return self.params

#-------------------------------------------------------------------------------
  def add_param(self, param_dict):
    if type(param_dict) is not dict:
      raise TypeError("Only dictionaries are accepted parameters.")
    self.params.append(param_dict)
    self.n_params = len(self.params)
    return self.ret_params()

#-------------------------------------------------------------------------------

