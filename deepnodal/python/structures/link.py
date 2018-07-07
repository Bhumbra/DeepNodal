"""
Base class for single input and output functionality to play nicely with TensorFlow.

It inherits from abstract class leaf. One leaf optionally supports a list of parameters,
stored as a list of ordered dictionaries in the form {'parameter_name': parameter_value}.

Each link is associated with with a creation which TensorFlow creations a `node' on the graph
which is not called until link.__call__() is invoked.
"""
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.leaf import *
from deepnodal.python.interfaces.calls import *

#-------------------------------------------------------------------------------
class link (leaf):
  """
  A link is a leaf connecting an input to an output via a creation. It has no
  hierarchical substructure (i.e. no subobjects).
  """
  # public 
  def_name = 'link'

  # protected 
  _inp = None
  _out = None
  _creation = None
  _args = None
  _kwds = None

  # private
  __var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    leaf.__init__(self, name, dev)
    self.set_creation()

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    self._creation = Creation(creation)
    self._args = args
    self._kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def __call__(self, inp = None, _called = True):
    inp = self.set_inp(inp)
    args = tuple(self._args)
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
    self.set_called(_called)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def _setup_params(self):
    self._params = []
    self._n_params = 0
    if self.__var_scope is None: return self._params
    for Param in list(Param_Dict):
      with Scope('var', self.__var_scope, reuse=True):
        try:
          param = Creation('ret_var')(Param)
        except ValueError:
          param = None
        if param is not None:
          self.add_param(mapping({self.__var_scope+"/"+Param_Dict[Param]: param}))
    self._n_params = len(self._params)
    return self.ret_params()

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = link()
    elif not isinstance(other, link) and not issubclass(other, link):
      raise TypeError("Cannot clone link to class " + str(other))

    # Change name and device...
    other.set_name(self.name)
    other.set_dev(self.dev)

    # ... before setting the creation in case this influences self.__var_scope
    other.set_creation(self._creation, *self._args, **self._kwds)
    return other

#-------------------------------------------------------------------------------

