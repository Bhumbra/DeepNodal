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

  The master class is abstract and inheriting classes must define self.__call__(inp)
  to be instantiated.
  """

  # public
  def_name = 'master'               # default name
  def_subworker = slave             # default subworker class
  def_subworker_name = 'subworker'  # default subworker name

  # protected
  _devs = None                      # slave devices
  _subworkers = None                # subworker instances which may be leaves or stems
  _n_subworkers = None              # number of subworkers
  _subworker = slave                # subworker class
  _unit_subworker = None            # flag to state unit subworker
  _subworker_name = 'subworker'     # default subworker name if not unit_subworker

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    self.set_name(name)
    self.set_dev(dev)
    self.set_subworker()
    self.set_subworkers()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self._subworkers is None: return
    for i, subworker in enumerate(self._subworkers):
    # no point re-naming subworkers if unit_subworker is true 
      subworker_name = self.name if self._unit_subworker else self.name + "/" + self._subworker_name + "_" + str(i)
      subworker.set_name(subworker_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None, devs = None):
    self.dev = dev
    self.devs = devs
    if self._subworkers is None or self.devs is None: return
    for _dev, subworker in zip(self.devs, self._subworkers):
      subworker.set_dev(_dev)

#-------------------------------------------------------------------------------
  def set_subworker(self, subworker = None, subworker_name = None):
    """
    This sets the class-type of each subworker and associated name prior to indexing.
    """
    self._subworker = subworker if subworker is not None else self.def_subworker
    self._subworker_name = subworker_name if subworker_name is not None else self.def_subworker_name

#-------------------------------------------------------------------------------
  def set_subworkers(self, subworkers = None):
    """
    This allows either manually setting the list of subworkers, or if subworkers is an
    integer it instantiates that number of subworkers.
    """
    self._subworkers = subworkers
    self._n_subworkers = 0
    self._unit_subworker = None
    if self._subworkers is None:
      return self._subworkers
    elif type(self._subworkers) is list:
      self._n_subworkers = len(self._subworkers)
      self._unit_subworker = self._n_subworkers == 1
      # it would be quite rude to rename or redevice these subworkers so we won't
    elif type(subworkers) is int:
      self._n_subworkers = subworkers
      self._unit_subworker = self._n_subworkers == 1
      self._subworkers = [self._subworker() for i in range(self._n_subworkers)]
      self.set_name(self.name) # this renames all subworkers
      self.set_dev(self.dev)   # this redevices all subworkers
    else:
      raise TypeError("Unrecognised subworkers specification.")
    return self._subworkers

#-------------------------------------------------------------------------------
  def __getitem__(self, index):
    return self._subworkers[index]

#-------------------------------------------------------------------------------
  def __len__(self):
    return self._n_subobjects

#-------------------------------------------------------------------------------
  @abstractmethod
  def __call__(self, inp = None): # this function is for calling graph workers
    pass

#-------------------------------------------------------------------------------

