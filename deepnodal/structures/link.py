"""
Base class for single input and output functionality to play nicely with TensorFlow.

It inherits from abstract class leaf. One leaf optinally supports a list of  parameters,
stored as a list of dictionaries in the form {'parameter_name', parameter_value}.

Each link is associated with with a creation which TensorFlow creations a `node' on the graph
which is not created until link.setup() is invoked.
"""
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.concepts.leaf import leaf
from deepnodal.calls.google_tensorflow import *

#-------------------------------------------------------------------------------
class link (leaf):
  """
  A link is a leaf connecting an input to an output via a creation. It has no
  hierarchical substructure (i.e. no subobjects).
  """
  def_name = 'link'
  out = None
  creation = None
  args = None
  kwds = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    leaf.__init__(self, name, dev)
    self.set_creation()
    self.setup()

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    self.creation = Creation[creation]
    self.args = args
    self.kwds = kwds

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    inp = self.set_inp(inp)
    if self.inp is not None and self.creation is not None:
      if self.dev is None:
        self.out = self.creation(self.inp, *self.args, **self.kwds)
      else:
        with Device(self.dev):
          self.out = self.creation(self.inp, *self.args, **self.kwds)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def set_params(self):
    self.params = []
    with Scope('var', self.name, reuse=True):
      for Param in Param_List:
        try:
          param = tf.get_variable(Param)
        except ValueError:
          param = None
      if param is not None:
        self.add_param({Param: param})
    self.n_params = len(self.params)
    return self.params

#-------------------------------------------------------------------------------

