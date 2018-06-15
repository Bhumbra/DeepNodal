# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class function (ABC):
  # public
  name = "function"
  dev = None

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self, name = None):
    self.name = name

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------

