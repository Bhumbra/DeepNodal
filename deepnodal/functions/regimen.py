"""
Regimen module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.functions.plan import *

#-------------------------------------------------------------------------------
class regimen (plan):
  """
  A regimen is a class that represents the atomic unit of training schedule.

  """
  def_name = 'regimen'
  gst = None            # global step
  dspec = None          # dropout specification
  pspec = None          # parameter specification

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    plan.__init__(name, dev)
    self.set_global_step()
    self.set_dropout_spec()
    self.set_parameter_spec()

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst

#-------------------------------------------------------------------------------
  def set_dropout_pspec(self, dspec = None):
    self.dspec = dspec

#-------------------------------------------------------------------------------
  def set_parameter_spec(self, pspec = None):
    self.pspec = pspec

#-------------------------------------------------------------------------------
  def setup(self, gstep = None): 
    # this creates the graph nodes for the learning rate
    if self.gstep is None: self.set_global_step(gstep)
    if self.lrate is not callable:
      if self.dev is None:
        self.learning_rate = Creation('var')(self.lrate, *self.lrate_args, **self.lrate_kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(self.lrate, *self.lrate_args, **self.lrate_kwds)
    else:
      if self.lrate == Creation('identity'):
        if self.dev is None:
          self.learning_rate = self.lrate(*self.lrate_args, **self.lrate_kwds)
        else:
          with Device(self.dev):
            self.learning_rate = self.lrate(*self.lrate_args, **self.lrate_kwds)
      else:
        if self.dev is None:
          self.learning_rate = self.lrate(*self.lrate_args, global_step = self.gstep, **self.lrate_kwds)
        else:
          with Device(self.dev):
            self.learning_rate = self.lrate(*self.lrate_args, global_step = self.gstep, **self.lrate_kwds)

#-------------------------------------------------------------------------------

