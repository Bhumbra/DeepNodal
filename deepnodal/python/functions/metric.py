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
  _test = None
  _train = None
  _dtypes = None
  _updater = None
  _summarise = None
  _scalar = None
  _scalars = None

  # private
  __var_scope = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    """ Regular initialisation """
    self.set_name(name)
    self.set_dev(dev)
    self.set_creation()
    self.set_dtypes()
    self.set_summarise()

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
  def set_summarise(self, summarise = False, train = None, test = None):
    """ Set whether to summarise and if data is train, test, both, or neither """
    self._summarise = summarise
    self._train = train
    self._test = test
    self._scalar = None
    self._scalars = {'train': None, 'test': None}

    if self.name is not None:
      if self._train is not None:
        self._train = False
      if self._test is not None:
        self._test = False
    else:
      name = self.name.split('/')[-1].lower()
      if self._train is None:
        self._train = 'train' in name
      if self._test is None:
        self._test = 'test' in name

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
    if not self._summarise: return None
    if out is None: return None
    if self._train is None or self._test is None:
      self.set_summarise(self._summarise, self._train, self._test)
    name = self.name.split('/')[0].lower()
    if self._train:
      scalar_name = self.name
      if 'train' not in scalar_name.lower():
        scalar_name += "_TRAIN"
      if self.dev is None:
        self._scalars['train'] = Summary('scalar')(scalar_name, out)
      else:
        with Device(self.dev):
          self._scalars['train'] = Summary('scalar')(scalar_name, out)
      self._scalar = self._scalars['train']
    if self._test:
      scalar_name = self.name
      if 'test' not in scalar_name.lower():
        scalar_name += "_TEST"
      if self.dev is None:
        self._scalars['test'] = Summary('scalar')(scalar_name, out)
      else:
        with Device(self.dev):
          self._scalars['test'] = Summary('scalar')(scalar_name, out)
      self._scalar = self._scalars['test']
    if self._train or self._test:
      return self._scalars
    scalar_name = self.name
    if self.dev is None:
      self._scalar  = Summary('scalar')(scalar_name, out)
    else:
      with Device(self.dev):
        self._scalar  = Summary('scalar')(scalar_name, out)
    return self._scalar

#-------------------------------------------------------------------------------
  def op_assign(self, target, reuse=False):
    with Scope('var', self.name+"_update", reuse=reuse):
      if self.dev is None:
        op = self._out.assign(target)
      else:
        with Device(self.dev):
          op = self._out.assign(target)
    return op

#-------------------------------------------------------------------------------
