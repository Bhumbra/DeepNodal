"""
Slave module for Tensorflow. The class slave is abstract, and inheriting classes
are only instantiable after defining self.setup.

"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.function import *

#-------------------------------------------------------------------------------
class slave (function):
  """
  A plan is a class that represents the conceptual training unit of a training 
  schedule. It is abstract and inheriting classes can only be instantiated when 
  self.__call__() is defined. The properties define the specification for the
  optimiser and learning rate, and progress.
  """

  # public
  def_name = 'slave'     # default name

  # protected
  _opt = None            # optimiser
  _lrn = None            # learning rate

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_optimiser()
    self.set_learning_rate()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_learning_rate(self, lrn = None, *lrn_args, **lrn_kwds):
    self._lrn = lrn
    self._lrn_args = lrn_args
    self._lrn_kwds = dict(lrn_kwds)

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = None, *opt_args, **opt_kwds):
    self._opt = opt
    self._opt_args = opt_args
    self._opt_kwds = dict(opt_kwds)

#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self): 
    pass

#-------------------------------------------------------------------------------

