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
  _creator = None
  _creator_args = None
  _creator_kwds = None
  _creator = None
  _creation = None
  _creation_args = None
  _creation_kwds = None
  _prototype = None
  _call_function = None
  _aux = None

  # private
  __var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name=None, dev=None):
    leaf.__init__(self, name, dev)
    self.set_creator()
    self.set_creation()

#-------------------------------------------------------------------------------
  def set_creator(self, creator=None, *args, **kwds):
    self._creator = Creation(creator)
    self._creator_args = args
    self._creator_kwds = dict(kwds)
 
#-------------------------------------------------------------------------------
  def set_creation(self, creation=None, *args, **kwds):
    self._creation = Creation(creation) if self._creator is None else creation
    self._creation_args = args
    self._creation_kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def __call__(self, inp=None, _called=True):
    self._call_creator()
    self._call_creation(inp)
    self.set_called(_called)
    self._map_params()
    self._map_moments()
    self._map_updates()
    return self.ret_out()

#-------------------------------------------------------------------------------
  def _call_creator(self):
    self._prototype = None
    if self._creator is None: return self._prototype
    args = tuple(self._creator_args)
    kwds = dict(self._creator_kwds)
    args, kwds = structuref2unique(*args, **kwds)
    self._prototype = self._creator(*args, **kwds)
    return self._prototype

#-------------------------------------------------------------------------------
  def _call_creation(self, inp=None):
    inp = self.set_inp(inp)
    args = tuple(self._creation_args)
    kwds = dict(self._creation_kwds)
    self.__var_scope = None
    if 'var_scope' in kwds:
      self.__var_scope = kwds['var_scope']
      kwds.pop('var_scope')
    elif 'name' in kwds:
      self.__var_scope = kwds['name']
    elif 'scope' in kwds:
      self.__var_scope = kwds['scope']
    if self._inp is None or self._creation is None: return self.ret_out()
    args, kwds = structuref2unique(*args, **kwds)
    self._call_function = self._creation
    if self._prototype is not None:
      self._call_function = getattr(self._prototype, self._creation)
    if 'var_scope' in self._creation_kwds:
      if self.dev is None:
        with Scope('var', self.__var_scope, reuse=Flag('auto_reuse')):
          self._out = self._call_function(self._inp, *args, **kwds)
      else:
        with Device(self.dev):
          with Scope('var', self.__var_scope, reuse=Flag('auto_reuse')):
            self._out = self._call_function(self._inp, *args, **kwds)
    else:
      if self.dev is None:
        self._out = self._call_function(self._inp, *args, **kwds)
      else:
        with Device(self.dev):
          self._out = self._call_function(self._inp, *args, **kwds)
    if isinstance(self._out, (list, tuple)):
      self._out, self._aux = self._out[0], self._out[1:]

#-------------------------------------------------------------------------------
  def _map_params(self):
    assert self._called, "Cannot setup params without object being called"
    self._params = []
    self._n_params = 0
    if self.__var_scope is None: return self._params

    # Add parameters belonging to architecture
    for Param in list(Param_Dict):
      with Scope('var', self.__var_scope, reuse=True):
        try:
          param = Creation('ret_var')(Param)
        except ValueError:
          param = None
        if param is not None:
          self.add_param(mapping({self.__var_scope+"/"+Param_Dict[Param]: param}))
     
    # Add parameters belonging to normalisation
    for Norm in list(Norm_Dict):
      with Scope('var', self.__var_scope, reuse=True):
        try:
          param = Creation('ret_var')(Norm)
        except ValueError:
          param = None
        if param is not None:
          self.add_param(mapping({self.__var_scope+"/"+Norm_Dict[Norm]: param}))

    # Add auxilliary parameters
    if self._aux:
      for i, aux in enumerate(self._aux):
        self.add_param(mapping({self.__var_scope+"/aux_param_"+str(i): aux}))

    return self._params

#-------------------------------------------------------------------------------
  def _map_moments(self):
    self._moments = []
    self._n_moments = 0
    for Moment in Norm_Moments:
      with Scope('var', self.__var_scope, reuse=True):
        try:
          moment = Creation('ret_var')(Moment)
        except ValueError:
          moment = None
        if moment is not None:
          self.add_moment(mapping({self.__var_scope+"/"+Moment: moment}))
    return self._moments

#-------------------------------------------------------------------------------
  def _map_updates(self):
    self._updates = []
    self._n_updates = 0
    updates = Updates(scope=self.name)
    for update in updates:
      if update is not None:
        self.add_update(mapping({self.name+"/"+str(update): update}))
    return self._updates

#-------------------------------------------------------------------------------
  def clone(self, other=None):
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
