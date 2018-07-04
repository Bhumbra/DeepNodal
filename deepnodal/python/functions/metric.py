"""
Base class for for input/output functionality to play nicely with TensorFlow.

It inherits from abstract class function. The metric class is the function-equivalent
of the structural base class link, except with the following differences:
  
  - there are no trainable parameters.
  - there are no hierarchies associated with metric evaluations.
  - metric objects are instantiated by classes inheriting from trainer.

"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.function import function
from deepnodal.python.interfaces.calls import *

#-------------------------------------------------------------------------------
class metric (function):
  """
  A metric is function with a defined input, function creation, arguments, and 
  keywords. It has no hierarchical substructure.
  """

  # public
  def_name = 'metric'

  # protected
  _inp = None
  _out = None
  _creation = None
  _args = None
  _kwds = None
  _var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    function.__init__(self, name, dev)
    self.set_creation()

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    self._creation = Creation(creation)
    self._args = args
    self._kwds = dict(kwds)

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
  def __call__(self, inp = None, _called = True):
    inp = self.set_inp(inp)
    kwds = dict(self._kwds)
    self._var_scope = None
    if 'var_scope' in kwds:
      self._var_scope = kwds['var_scope']
      kwds.pop('var_scope')
    elif 'name' in kwds:
      self._var_scope = self._kwds['name']
    elif 'scope' in self._kwds:
      self._var_scope = self._kwds['scope']
    if self._inp is None or self._creation is None: return self.ret_out()
    if 'var_scope' in self._kwds:
      if self.dev is None:
        with Scope('var', self._var_scope, reuse=Flag('auto_reuse')):
          self._out = self._creation(self._inp, *self._args, **kwds)
      else:
        with Device(self.dev):
          with Scope('var', self._var_scope, reuse=Flag('auto_reuse')):
            self._out = self._creation(self._inp, *self._args, **kwds)
    else:
      if self.dev is None:
        self._out = self._creation(self._inp, *self._args, **self._kwds)
      else:
        with Device(self.dev):
          self._out = self._creation(self._inp, *self._args, **self._kwds)
    self.set_called(_called)
    return self.ret_out()

#-------------------------------------------------------------------------------

