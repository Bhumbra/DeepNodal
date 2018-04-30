"""
Hypervisor module for Tensorflow. 
"""
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.supervisor import *

#------------------------------------------------------------------------------- 
class hypervisor (supervisor, master, stem):
  """
  A hypervisor is a class that distributes computations for supervised 
  learning across multiple devices.

  It is three things in one class:

  A supervisor: it updates parameters according to gradient-based optimisation.
  A master: it supervises supervisors that evaluate the gradients.
  A stem: it is effectively the final `trunk', that clones network specifications.

  From coders' perspectives, it is called exactly the same way as the supervisor
  except at it's instantiation:

  sup = supervisor() and sup = hypervisor() are identical

  sup = supervisor('name', 'dev') and sup = hypervisor('name', 'dev') are identical.

  sup = hypervisor('name', 2) however sets up a hypervisor instance that distributes
  supervised learning across 2 GPU devices.

  """
  def_name = 'hypervisor'

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    supervisor.__init__(self, name, dev)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev

#-------------------------------------------------------------------------------

