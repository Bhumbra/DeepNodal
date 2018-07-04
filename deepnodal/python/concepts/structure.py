# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class structure (ABC):
  # public
  name = "structure"
  dev = None

  # protected
  _called = False
  _parent = None

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self, name = None):
    self.name = name

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------
  def set_parent(self, _parent = None):
    self._parent = _parent

#-------------------------------------------------------------------------------
  def ret_parent(self):
    return self._parent

#-------------------------------------------------------------------------------
  def set_called(self, _called = False):
    self._called = _called

#-------------------------------------------------------------------------------
  def ret_called(self):
    return self._called

#-------------------------------------------------------------------------------
