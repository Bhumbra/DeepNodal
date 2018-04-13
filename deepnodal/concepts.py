# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class structure (ABC):
  @abstractmethod
  def set_name(self):
    pass
  def set_dev(self):
    pass

#-------------------------------------------------------------------------------
class function (ABC):
  @abstractmethod
  def set_name(self):
    pass
  def set_dev(self):
    pass

#-------------------------------------------------------------------------------
