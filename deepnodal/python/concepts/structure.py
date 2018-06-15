# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class structure (ABC):
  # public
  name = "structure"
  dev = None

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self, name = None):
    self.name = name

#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self, name = None):
    self.dev = dev

#-------------------------------------------------------------------------------
