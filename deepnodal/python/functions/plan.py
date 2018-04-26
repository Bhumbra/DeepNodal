"""
Plan module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.function import *
from deepnodal.python.calls.google_tensorflow import *

#-------------------------------------------------------------------------------
class plan (function):
  """
  A plan is a class that represents the conceptual training unit of a training 
  schedule. It is abstract and inheriting classes can only be instantiated when 
  self.setup is defined. The most basic example of this is the class regimen.
  """

  def_name = 'plan'
  lrate = None          # learning rate

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_learning_rate()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_learning_rate(self, lrate = None, *lrate_args, **lrate_kwds):
    self.lrate = Creation(lrate)
    self.lrate_args = lrate_args
    self.lrate_kwds = dict(lrate_kwds)

#-------------------------------------------------------------------------------
  @abstractmethod
  def setup(self): 
    pass

#-------------------------------------------------------------------------------

