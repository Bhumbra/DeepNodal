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
  self.setup is defined. The properties define the specification for the
  optimiser and learning rate, and progress.

  """

  def_slave = 'status'
  opt = None            # optimiser
  lrn = None            # learning rate
  progress = None       # progress list: [number of batch_updates, sum(batch_sizes)]

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_optimiser()
    self.set_learning_rate()
    self.set_progress()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_learning_rate(self, lrn = None, *lrn_args, **lrn_kwds):
    self.lrn = lrn
    self.lrn_args = lrn_args
    self.lrn_kwds = dict(lrn_kwds)

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = None, *opt_args, **opt_kwds):
    self.opt = opt
    self.opt_args = opt_args
    self.opt_kwds = dict(opt_kwds)

#-------------------------------------------------------------------------------
  def set_progress(self, progress = None):
    self.progress = progress
    if self.progress is None:
      self.progress = [-1, 0] # [number of batch_updates, sum(batch_sizes)]
    return self.progress

#-------------------------------------------------------------------------------
  @abstractmethod
  def setup(self): 
    pass

#-------------------------------------------------------------------------------

