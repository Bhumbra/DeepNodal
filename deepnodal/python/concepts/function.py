# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class function (ABC):
  name = "function"
  dev = None
#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self):
    self.name = name
#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self):
    self.dev = dev

#-------------------------------------------------------------------------------

