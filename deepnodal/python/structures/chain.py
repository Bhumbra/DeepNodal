# Base class for single input and output functionality through a single stream.
#
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import *
from deepnodal.python.structures.link import *

#-------------------------------------------------------------------------------
class chain (stem):
  """
  A chain is a `stem' structure with of one or many links in series.
  While chain is not abstract, it is not intended to instantiated directly
  since it has no parameter collation facility: see stream.

  A chain is the simplest class exhibiting a self.outputs list of dictionaries
  containing the single mapping of [{output_name: output.object}]
  """
  def_name = 'chain'
  def_subobject_name = 'link'
  def_subobject = link
  inp = None
  out = None
  links = None
  n_links = None
  unit_link = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    # We add links sequentially before declaring as subobjects so
    # the stem version of this function must be overloaded here...

    self.name = name if name is not None else self.def_name
    if self.links is None: return
    for i, _link in enumerate(self.links):
      # no point re-naming links if unit_link true 
      link_name = self.name if self.unit_link else self.name + "/" + self.subobject_name + "_" + str(i)
      _link.set_name(link_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    # We add links sequentially before declaring as subobjects so
    # the stem version of this function must be overloaded here...
    self.dev = dev
    if self.links is None: return
    for _link in self.links:
      _link.set_dev(dev)

#-------------------------------------------------------------------------------
  def add_link(self, creation = None, *args, **kwds):
    # this requires no input since it does not create graph objects
    if self.links is None:
      self.links = []
      self.n_links = 0
      self.unit_link = False
    if creation is None:
      return
    if isinstance(creation, link) and not(len(args)) and not(len(kwds)):
      self.links.append(creation)
    else:
      if 'name' in kwds:
        name = kwds['name']
      elif 'scope' in kwds:
        name = kwds['scope']
      else:
        name = 'link_' + str(self.n_links)
        if self.name is not None:
          name = self.name + "/" + name
      self.links.append(link(name, self.dev))
    self.n_links = len(self.links)
    self.unit_link = self.n_links == 1
    self.links[-1].set_creation(creation, *args, **kwds)
    return self.links[-1]

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    self.inp = inp
    self.out = None
    if self.links is None: return
    for _link in self.links:
      inp = _link.setup(inp)
    self.out = inp
    return self.ret_out()

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    # Create chain instance if necessary
    if other is None:
      other = chain()
    elif not isinstance(other, chain) and not issubclass(other, chain):
      raise TypeError("Cannot clone chain to class " + str(other))
    elif other.links is not None:
      raise AttributeError("Cannot clone to a chain instance with pre-existing links")

    # All we have to do now is clone the links
    if self.links is not None:
      for _link in self.links:
        other.add_link(_link.clone())

    # Now rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)
    return other

#-------------------------------------------------------------------------------

