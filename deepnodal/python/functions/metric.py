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
  _dtypes = None
  _updater = None
  _label = None
  _scalar = None
  _scalars = None

  # private
  __var_scope = None
  __deimiter = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    """ Regular initialisation """
    self.set_name(name)
    self.set_dev(dev)
    self.set_creation()
    self.set_dtypes()
    self.set_label()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_creation(self, creation = None, *args, **kwds):
    """ Allow metric to be a custom-called value """
    self._creation = Creation(creation)
    self._args = args
    self._kwds = dict(kwds)
      
#-------------------------------------------------------------------------------
  def set_dtypes(self, *dtypes):
    """ Set data types of custom-called arguments """
    self._dtypes = dtypes if len(dtypes) else None
    if self._dtypes is None: return
    for arg, dtype in zip(self._args, self._dtypes):
      if dtype is not None:
        if type(arg) is not list:
          raise TypeError("Any dtype specification must relate to a list argument.")

#-------------------------------------------------------------------------------
  def set_label(self, label = None, *args, delimiter='/'):
    """ Set metric label according the keys in args (default empty) """
    self._label = label
    keys = tuple(args)
    self.__delimiter = delimiter
    self._scalar = None
    self._scalars = {}
    if keys:
      for key in keys:
        self._scalars.update({key: None})
    if self._label is None:
      self._label = 'METRIC'
    if self.name:
      self._label = self.name + "/" + self._label

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
    if not(self._called): 
      raise ValueError("Not implemented for when not called.")
      return self, metric.ret_out
    return self._out

#-------------------------------------------------------------------------------
  def ret_scalar(self, spec=None):
    if not(self._called): return self, metric.ret_scalar, spec
    if spec is None: return self._scalar
    if spec not in self._scalars:
      return None
    return self._scalars[spec]

#-------------------------------------------------------------------------------
  def ret_label(self, spec=None):
    """ Returns the (label, sublabel) tuple for spec"""
    delim = self.__delimiter
    label = self._label
    sublabel = label.split(delim)[-1]
    if spec is None or len(self._scalars) < 2:
      return label, sublabel
    if spec.lower() in sublabel.lower():
      return label, sublabel
    label += "_{}".format(spec.upper())
    sublabel += "_{}".format(spec.upper())
    return label, sublabel

#-------------------------------------------------------------------------------
  def __call__(self, inp = None, _called = True):
    if inp is not None:
      inp = self.set_inp(inp)
    self._out = self._inp
    if self._creation:
      args, kwds = structuref2unique(*self._args, **self._kwds)
      self.__var_scope = None
      if 'var_scope' in kwds:
        self.__var_scope = kwds['var_scope']
        kwds.pop('var_scope')
      elif 'name' in kwds:
        self.__var_scope = self._kwds['name']
      elif 'scope' in self._kwds:
        self.__var_scope = self._kwds['scope']
      if self.dev is None:
        if self._inp is None:
          self._out = self._creation(*args, **kwds)
        else:
          self._out = self._creation(self._inp, *args, **kwds)
      else:
        with Device(self.dev):
          if self._inp is None:
            self._out = self._creation(*args, **kwds)
          else:
            self._out = self._creation(self._inp, *args, **kwds)
    self.__call__scalars(self._out)
    self.set_called(True)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def __call__scalars(self, out = None):
    if not self._label: return None
    if out is None: return None

    # Handle trivial case first
    if not len(self._scalars):
      if self.dev is None:
        self._scalar  = Summary('scalar')(label, out)
      else:
        with Device(self.dev):
          self._scalar  = Summary('scalar')(label, out)
      return self._scalar
    
    # Handle multiple key method:
    label_lower = self._label.split(self.__delimiter)[0].lower()
    keys = list(self._scalars.keys())
    for key in keys:
      label = self._label
      if len(keys) > 1:
        if key.lower() not in label_lower:
          label += "_" + key.upper()
      if self.dev is None:
        self._scalars[key] = Summary('scalar')(label, out)
      else:
        with Device(self.dev):
          self._scalars[key] = Summary('scalar')(label, out)
    if len(keys) == 1:
      self._scalar = self._scalars[key]
    return self._scalars

#-------------------------------------------------------------------------------
  def call_assign(self, target, reuse=False):
    with Scope('var', self.name+"_update", reuse=reuse):
      if self.dev is None:
        op = self._out.assign(target)
      else:
        with Device(self.dev):
          op = self._out.assign(target)
    return op

#-------------------------------------------------------------------------------
