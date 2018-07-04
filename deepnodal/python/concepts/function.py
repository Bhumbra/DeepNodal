# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class function (ABC):
  # public
  name = "function"
  dev = None

  # protected
  _called = False

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self, name = None):
    self.name = name

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_called(self, _called = False):
    self._called = _called

#-------------------------------------------------------------------------------
  def ret_called(self):
    return self._called

#-------------------------------------------------------------------------------
