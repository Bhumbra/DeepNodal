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
from deepnodal.python.concepts.leaf import *
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
  _inputs = None
  _outputs = None
  _test = False
  _train = False
  _dtypes = None
  _updater = None
  _summary = None
  _summarise = None

  # private
  __var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    function.__init__(self, name, dev)
    self.set_creation()
    self.set_dtypes()
    self.set_summary()

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    self._creation = Creation(creation)
    self._args = args
    self._kwds = dict(kwds)
      
#-------------------------------------------------------------------------------
  def set_dtypes(self, *dtypes):
    self._dtypes = dtypes if len(dtypes) else None
    if self._dtypes is None: return
    for arg, dtype in zip(self._args, self._dtypes):
      if dtype is not None:
        if type(arg) is not list:
          raise TypeError("Any dtype specification must relate to a list argument.")

#-------------------------------------------------------------------------------
  def set_summary(self, summarise = False, train = False, test = False):
    self._summarise = summarise
    self._train = train
    self._test = test

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
    if not(self._called): return self, metric.ret_out
    return self._out

#-------------------------------------------------------------------------------
  def set_called(self, _called):
    self._called = _called

#-------------------------------------------------------------------------------
  def ret_called(self):
    return self._called

#-------------------------------------------------------------------------------
  def __call__(self, inp = None, _called = True):
    if inp is not None:
      inp = self.set_inp(inp)
    elif self._inputs is None:
      raise ValueError("Cannot call metric without specification of inputs.")
    elif not(len(self._inputs_args)) and not(len(self._inputs__kwds)):
      inp = structuref2unique((inpt))[0][0]
      inp = self.set_inp(inp)
    kwds = dict(self._kwds)
    self.__var_scope = None
    if 'var_scope' in kwds:
      self.__var_scope = kwds['var_scope']
      kwds.pop('var_scope')
    elif 'name' in kwds:
      self.__var_scope = self._kwds['name']
    elif 'scope' in self._kwds:
      self.__var_scope = self._kwds['scope']
    if self._inp is None or self._creation is None: return self.ret_out()
    args, kwds = structuref2unique(*args, **kwds)
    if self.dev is None:
      self._out = self._creation(self._inp, *args, **kwds)
    else:
      with Device(self.dev):
        self._out = self._creation(self._inp, *args, **kwds)
    self.__call__summaries()
    self._called = True
    return self.ret_out()

#-------------------------------------------------------------------------------
  def __call__summaries(self):
    pass

#-------------------------------------------------------------------------------
  def op_assignout(self, target):
    op = self._out.assign(target)
    return op

#-------------------------------------------------------------------------------

