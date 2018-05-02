"""
Regime module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.slave import *
from deepnodal.python.interfaces.calls import *

#-------------------------------------------------------------------------------
class regime (slave):
  """
  A regime is a slave with a learning rate specification but without optimiser
  and progress treatment. Instead regime provides individual treatment for
  dropout and parameter specifications.
  """
  def_name = 'regime'
  gst = None            # global step
  dro = None            # dropout specification
  par = None            # parameter specification
  learning_rate = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    slave.__init__(self, name, dev)
    self.set_global_step()
    self.set_dropouts()
    self.set_parameters()

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst

#-------------------------------------------------------------------------------
  def set_dropouts(self, dro = None):
    self.dro = dro

#-------------------------------------------------------------------------------
  def set_parameters(self, par = None):
    self.par = par

#-------------------------------------------------------------------------------
  def setup(self, gst = None): 
    # this creates the graph nodes for the learning rate
    if self.gst is None and gst is not None: self.set_global_step(gst)

    lrn = Creation(self.lrn)
    kwds = dict(self.lrn_kwds)
    if 'name' not in kwds:
      kwds.update({'name': self.name + '/learning_rate'})

    if not callable(lrn):
      if self.dev is None:
        self.learning_rate = Creation('var')(lrn, *self.lrn_args, **kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(lrn, *self.lrn_args, **kwds)
    else:
      if lrn == Creation('var'):
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, dtype=Dtype('float32'), **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, dtype=Dtype('float32'), **kwds)
      elif lrn == Creation('identity'):
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, **kwds)
      else:
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, global_step = self.gst, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, global_step = self.gst, **kwds)
    return self.learning_rate


#-------------------------------------------------------------------------------

