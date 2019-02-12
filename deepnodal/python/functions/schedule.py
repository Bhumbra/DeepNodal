"""
Schedule module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.slave import *
from deepnodal.python.interfaces.calls import *

#-------------------------------------------------------------------------------
class schedule (slave):
  """
  A schedule is a slave with a learning rate specification but without optimiser
  and progress treatment. Instead schedule provides individual treatment for
  dropout and parameter specifications.
  """
  def_name = 'schedule'
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
  def __call__(self, gst = None): 
    # this creates the graph nodes for the learning rate
    if self.gst is None and gst is not None: self.set_global_step(gst)

    lrn = Creation(self._lrn)
    args = self._lrn_args
    kwds = dict(self._lrn_kwds)
    if 'name' not in kwds:
      kwds.update({'name': self.name + '/learning_rate'})

    if not callable(lrn):
      if self.dev is None:
        self.learning_rate = Creation('var')(lrn, *args, **kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(lrn, *args, **kwds)
    else:
      if lrn == Creation('var'):
        if self.dev is None:
          self.learning_rate = lrn(*args, dtype=Dtype('float32'), **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*args, dtype=Dtype('float32'), **kwds)
      elif lrn == Creation('identity'):
        if self.dev is None:
          self.learning_rate = lrn(*args, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*args, **kwds)
      else:
        if self.dev is None:
          self.learning_rate = lrn(*args, global_step = self.gst, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*args, global_step = self.gst, **kwds)
    self.set_called(True)
    return self.learning_rate

#-------------------------------------------------------------------------------
  def ret_lrate(self):
    if not self.ret_called(): return self, self.ret_lrate
    return self.learning_rate

#-------------------------------------------------------------------------------

