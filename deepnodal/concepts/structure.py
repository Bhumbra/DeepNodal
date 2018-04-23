# Abstract classes to inter-relate TensorFlow objects nicely

# Gary Bhumbra

#-------------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------------
class structure (ABC):
  name = "structure"
  dev = None
#-------------------------------------------------------------------------------
  @abstractmethod
  def set_name(self):
    pass
#-------------------------------------------------------------------------------
  @abstractmethod
  def set_dev(self):
    pass
#-------------------------------------------------------------------------------

