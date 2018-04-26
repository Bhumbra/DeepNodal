"""
Regimen module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.plan import *

#-------------------------------------------------------------------------------
class regimen (plan):
  """
  A regimen is a class that represents the atomic unit of training schedule.

  """
  def_name = 'regimen'
  gst = None            # global step
  dspec = None          # dropout specification
  pspec = None          # parameter specification
  learning_rate = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    plan.__init__(self, name, dev)
    self.set_global_step()
    self.set_dropout_spec()
    self.set_parameter_spec()

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst

#-------------------------------------------------------------------------------
  def set_dropout_spec(self, dspec = None):
    self.dspec = dspec

#-------------------------------------------------------------------------------
  def set_parameter_spec(self, pspec = None):
    self.pspec = pspec

#-------------------------------------------------------------------------------
  def setup(self, gst = None): 
    # this creates the graph nodes for the learning rate
    if self.gst is None and gst is not None: self.set_global_step(gst)
    if self.lrate is not callable:
      if self.dev is None:
        self.learning_rate = Creation('var')(self.lrate, *self.lrate_args, name = self.name+'/learning_rate', **self.lrate_kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(self.lrate, *self.lrate_args, name = self.name+'/learning_rate', **self.lrate_kwds)
    else:
      if self.lrate == Creation('identity'):
        if self.dev is None:
          self.learning_rate = self.lrate(*self.lrate_args, name = self.name+'/learning_rate', **self.lrate_kwds)
        else:
          with Device(self.dev):
            self.learning_rate = self.lrate(*self.lrate_args, name = self.name+'/learning_rate', **self.lrate_kwds)
      else:
        if self.dev is None:
          self.learning_rate = self.lrate(*self.lrate_args, global_step = self.gstep, name = self.name+'/learning_rate', **self.lrate_kwds)
        else:
          with Device(self.dev):
            self.learning_rate = self.lrate(*self.lrate_args, global_step = self.gstep, name = self.name+'/learning_rate', **self.lrate_kwds)

#-------------------------------------------------------------------------------

