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
from deepnodal.python.functions.accumulator import Accumulator
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
  _accum = None
  _group_names = None
  _groups = None

  # private
  __var_scope = None
  __delimiter = None
  __accumulators = None

#-------------------------------------------------------------------------------
  def __init__(self, name=None, dev=None):
    """ Regular initialisation """
    self.set_name(name)
    self.set_dev(dev)
    self.set_creation()
    self.set_dtypes()
    self.set_label()
    self.set_groups()

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    self.name = name

#-------------------------------------------------------------------------------
  def set_dev(self, dev=None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_creation(self, creation=None, *args, **kwds):
    """ Allow metric to be a custom-called value """
    self._creation = Creation(creation)
    self._args = tuple(args)
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
  def set_label(self, label=None, delimiter='/'):
    """ Set metric label according the keys in args (default empty) """
    self._label = label
    self.__delimiter = delimiter
    if self._label is None:
      self._label = 'METRIC'
    if self.name:
      self._label = self.name + "/" + self._label

#-------------------------------------------------------------------------------
  def set_groups(self, *args):
    """ Set metric sublabels according the keys in args (default empty) """
    self._scalar = None
    if len(args) == 1 and isinstance(args[0], dict):
      self._groups = args[0]
      self._group_names = list(args[0].keys())
    else:
      self._group_names = list(args)
      self._groups = {group_name: 0 for group_name in self._group_names}
    self._scalars = {}
    if self._group_names:
      for group_name in self._group_names:
        self._scalars.update({group_name: None})

#-------------------------------------------------------------------------------
  def set_inp(self, inp=None):
    self._inp = inp
    self._out = None
    return self._inp

#-------------------------------------------------------------------------------
  def ret_inp(self):
    return self._inp
  
#-------------------------------------------------------------------------------
  def ret_out(self):
    if not self._called: 
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
    label = self._label
    sublabel = label.split(self.__delimiter)[-1]
    if spec is None or len(self._scalars) < 2:
      return label, sublabel
    if spec.lower() in sublabel.lower():
      return label, sublabel
    label += "_{}".format(spec.upper())
    sublabel += "_{}".format(spec.upper())
    return label, sublabel

#-------------------------------------------------------------------------------
  def __call__(self, inp=None, _called=True):
    if inp is not None:
      inp = self.set_inp(inp)
      if isinstance(self._inp, str):
        assert self._creation is None, \
            "Calling metric with name incompatible with creation {}".format(
             self._creation)
        self._out = Creation('var')(0., name=self._inp)
        self.__call__scalars(self._out)
        self.set_called(_called)
        return self.ret_out()
    self._out = self._inp
    args, kwds = structuref2unique(*self._args, **self._kwds)
    if self._inp is not None:
      args = tuple([self._inp] + list(args))
    self.__var_scope = None
    if 'var_scope' in kwds:
      self.__var_scope = kwds['var_scope']
      kwds.pop('var_scope')
    elif 'name' in kwds:
      self.__var_scope = kwds['name']
    elif 'scope' in kwds:
      self.__var_scope = kwds['scope']
    if self._creation is None:
      if self._inp is not None:
        self._out = self._inp
    else:
      if 'var_scope' in self._kwds:
        if self.dev is None:
          with Scope('var', self.__var_scope, reuse=Flag('auto_reuse')):
            self._out = self._creation(*args, **kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.__var_scope, reuse=Flag('auto_reuse')):
              self._out = self._creation(*args, **kwds)
      else:
        if self.dev is None:
          self._out = self._creation(*args, **kwds)
        else:
          with Device(self.dev):
            self._out = self._creation(*args, **kwds)
    self._accum_out = self.__call__accumulators(self._out)
    self.__call__scalars(self._accum_out)
    self.set_called(_called)
    return self.ret_out()

#-------------------------------------------------------------------------------
  def __call__accumulators(self, out=None):
    if max(list(self._groups.values())) == 0:
      return out
    self.__accumulators = {group_name: None for group_name in self._group_names}
    accum_out = {group_name: out for group_name in self._group_names}
    for group_name, group_dsize in self._groups.items():
      if group_dsize:
        self.__accumulators[group_name] = Accumulator(
            self._label.lower() + "/" + group_name, self.dev)
        self.__accumulators[group_name].set_dsize(group_dsize)
        accum_out[group_name] = self.__accumulators[group_name].__call__(
                                  accum_out[group_name])
    return accum_out

#-------------------------------------------------------------------------------
  def __call__scalars(self, out=None):
    if out is None: return None

    # Handle trivial case first
    if not len(self._scalars):
      assert not isinstance(out, dict), \
          "Group specification required if using  accumulators"                                         
      label = self._label
      if self.dev is None:
        self._scalar = Summary('scalar')(label, out)
      else:
        with Device(self.dev):
          self._scalar  = Summary('scalar')(label, out)
      return self._scalar
    
    # Handle multiple key method:
    label_lower = self._label.split(self.__delimiter)[-1].lower()
    keys = list(self._scalars.keys())
    for key in keys:
      label = self._label
      group_out = out if not isinstance(out, dict) else out[key] 
      if len(keys) > 1:
        if key.lower() not in label_lower:
          label += "_" + key.upper()
      if self.dev is None:
        self._scalars[key] = Summary('scalar')(label, group_out)
      else:
        with Device(self.dev):
          self._scalars[key] = Summary('scalar')(label, group_out)
    if len(keys) == 1: self._scalar = self._scalars[key]
    return self._scalars

#-------------------------------------------------------------------------------
  def ret_accumulator(self, group=None):
    if self.__accumulators is None:
      return None
    if group is None:
      return self.__accumulators
    return self.__accumulators[group]

#-------------------------------------------------------------------------------
  def ret_accum_out(self, group=None):
    if self.__accumulators is None:
      return self.ret_out()
    if group is None:
      return self._accum_out
    return self._accum_out[group]

#-------------------------------------------------------------------------------
  def assign_op(self, value):
    return self._out.assign(value)

#-------------------------------------------------------------------------------
