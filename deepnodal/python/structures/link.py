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
from deepnodal.python.concepts.leaf import leaf
from deepnodal.python.interfaces.calls import *

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
  var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    leaf.__init__(self, name, dev)
    self.set_creation()
    self.setup()

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    self.creation = Creation(creation)
    self.args = args
    self.kwds = dict(kwds)

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    inp = self.set_inp(inp)
    kwds = dict(self.kwds)
    self.var_scope = None
    if 'var_scope' in kwds:
      self.var_scope = kwds['var_scope']
      kwds.pop('var_scope')
    elif 'name' in kwds:
      self.var_scope = self.kwds['name']
    elif 'scope' in self.kwds:
      self.var_scope = self.kwds['scope']
    if self.inp is None or self.creation is None: return self.ret_out()
    if 'var_scope' in self.kwds:
      if self.dev is None:
        with Scope('var', self.var_scope, reuse=Flag('auto_reuse')):
          self.out = self.creation(self.inp, *self.args, **kwds)
      else:
        with Device(self.dev):
          with Scope('var', self.var_scope, reuse=Flag('auto_reuse')):
            self.out = self.creation(self.inp, *self.args, **kwds)
    else:
      if self.dev is None:
        self.out = self.creation(self.inp, *self.args, **self.kwds)
      else:
        with Device(self.dev):
          self.out = self.creation(self.inp, *self.args, **self.kwds)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def setup_params(self):
    self.params = []
    self.n_params = 0
    if self.var_scope is None: return self.params
    for Param in list(Param_Dict):
      with Scope('var', self.var_scope, reuse=True):
        try:
          param = tf.get_variable(Param)
        except ValueError:
          param = None
        if param is not None:
          self.add_param({self.var_scope+"/"+Param_Dict[Param]: param})
    self.n_params = len(self.params)
    return self.params

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    if other is None:
      other = link()
    elif not isinstance(other, link) and not issubclass(other, link):
      raise TypeError("Cannot clone link to class " + str(other))

    # Change name and device...
    other.set_name(self.name)
    other.set_dev(self.dev)

    # ... before setting the creation in case this influences self.var_scope
    other.set_creation(self.creation, *self.args, **self.kwds)
    return other

#-------------------------------------------------------------------------------

