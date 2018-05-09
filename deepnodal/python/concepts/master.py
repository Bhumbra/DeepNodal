# Master module for Tensorflow. A master is a function that controls one or more
# subfunctions called subworkers. These subworkers may be classes that inherit 
# from master or from slave. 

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.slave import *

#-------------------------------------------------------------------------------
# We inherit function because a master does not require the same functionality as a slave
class master (function): 
  """
  A master is a function that supports and broadcasts to many subfunctions called
  subworkers. Note a subworkers may be another stem or a leaf (the default).

  The master class is abstract and inheriting classes must define self.setup(inp)
  to be instantiated.
  """

  def_name = 'master'              # default name
  devs = None                      # slave devices
  def_subworker = slave            # default subworker class
  def_subworker_name = 'subworker' # default subworker name
  subworkers = None                # subworker instances which may be leaves or stems
  n_subworkers = None              # number of subworkers
  subworker = slave                # subworker class
  unit_subworker = None            # flag to state unit subworker
  subworker_name = 'subworker'     # default subworker name if not unit_subworker

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_subworker()
    self.set_subworkers()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self.subworkers is None: return
    for i, subworker in enumerate(self.subworkers):
    # no point re-naming subworkers if unit_subworker is true 
      subworker_name = self.name if self.unit_subworker else self.name + "/" + self.subworker_name + "_" + str(i)
      subworker.set_name(subworker_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None, devs = None):
    self.dev = dev
    self.devs = devs
    if self.subworkers is None or self.devs is None: return
    for _dev, subworker in zip(self.devs, self.subworkers):
      subworker.set_dev(_dev)

#-------------------------------------------------------------------------------
  def set_subworker(self, subworker = None, subworker_name = None):
    """
    This sets the class-type of each subworker and associated name prior to indexing.
    """
    self.subworker = subworker if subworker is not None else self.def_subworker
    self.subworker_name = subworker_name if subworker_name is not None else self.def_subworker_name

#-------------------------------------------------------------------------------
  def set_subworkers(self, subworkers = None):
    """
    This allows either manually setting the list of subworkers, or if subworkers is an
    integer it instantiates that number of subworkers.
    """
    self.subworkers = subworkers
    self.n_subworkers = 0
    self.unit_subworker = None
    if self.subworkers is None:
      return self.subworkers
    elif type(self.subworkers) is list:
      self.n_subworkers = len(self.subworkers)
      self.unit_subworker = self.n_subworkers == 1
      # it would be quite rude to rename or redevice these subworkers so we won't
    elif type(subworkers) is int:
      self.n_subworkers = subworkers
      self.unit_subworker = self.n_subworkers == 1
      self.subworkers = [self.subworker() for i in range(self.n_subworkers)]
      self.set_name(self.name) # this renames all subworkers
      self.set_dev(self.dev)   # this redevices all subworkers
    else:
      raise TypeError("Unrecognised subworkers specification.")
    return self.subworkers

#-------------------------------------------------------------------------------
  @abstractmethod
  def setup(self, inp = None): # this function is for creating graph workers
    pass

#-------------------------------------------------------------------------------

