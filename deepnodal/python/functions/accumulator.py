"""
Class to provide local caching with accumulation to circumvent TensorFlow's 
slow-down and memory leaks with repeated tf.Variable.assign(datum) calls.
"""
DEFAULT_DATA_TYPE = 'float32'
DEFAULT_CACHE_SIZE = 200

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.function import function
from deepnodal.python.interfaces.calls import *
from tensorflow.python.ops import array_ops

#-------------------------------------------------------------------------------
class Accumulator (function):

  # Public
  def_name = 'accumulator'

  # Protected
  _dtype = None
  _dsize = None
  _cache = None
  _next = None
  _prev = None

#-------------------------------------------------------------------------------
  def __init__(self, name=None, dev=None):
    """ Regular initialisation """
    self.set_name(name)
    self.set_dev(dev)

#-------------------------------------------------------------------------------
  def set_name(self, name=None):
    self.name = name

#-------------------------------------------------------------------------------
  def set_dev(self, dev=None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_dtype(self, dtype=DEFAULT_DATA_TYPE):
    """ Set data types of custom-called arguments """
    self._dtype = dtype_dict(dtype)
    return self._dtype

#-------------------------------------------------------------------------------
  def set_dsize(self, dsize=DEFAULT_CACHE_SIZE):
    self._dsize = dsize

#-------------------------------------------------------------------------------
  def __call__(self, inp, _called=True):
    if self.dev is None:
      self._cache = Creation('var')([0.] * self._dsize, 
                                    dtype=self._dtype, 
                                    trainable=False,
                                    shape = [self._dsize])
      self._next = Creation('var')(0, dtype=dtype_dict['int32'])
      self._prev = Creation('var')(0, dtype=dtype_dict['int32'])
      self._assign_prev = self._prev.assign(self._next)
      self._last = self._cache[self._prev]
      self._assign_last = self._last.assign(inp)
      self._update_ops = Creation('combine')(self._assign_last, self._assign_prev)
      with Creation('deps')([self._update_ops]):
        self._incr_next = self._next.assign_add(1)
      with Creation('deps')([self._update_ops]):
        self._zero_next = self._next.assign(0)
      self._used_cache = Creation('slice')(self._cache, [0], [self._prev])
      self._average = Creation('mean')(self._used_cache)
    else:
      with Device(self.dev):
        self._next = Creation('var')(0, dtype=dtype_dict['int32'])
        self._prev = Creation('var')(0, dtype=dtype_dict['int32'])
        self._assign_prev = self._prev.assign(self._next)
        self._last = self._cache[self._prev]
        self._assign_last = self._last.assign(inp)
        self._update_ops = Creation('combine')(self._assign_last, self._assign_prev)
        with Creation('deps')([self._update_ops]):
          self._incr_next = self._next.assign_add(1)
        with Creation('deps')([self._update_ops]):
          self._zero_next = self._next.assign(0)
        self._used_cache = Creation('slice')(self._cache, [0], [self._prev])
        self._average = Creation('mean')(self._used_cache)
    return self._average

#-------------------------------------------------------------------------------
  def ret_average(self):
    return self._average

#-------------------------------------------------------------------------------
  def ret_update_ops(self, reset=None):
    if reset is None:
      return self._update_ops
    elif reset is False:
      return self._incr_next
    return self._zero_next

#-------------------------------------------------------------------------------
