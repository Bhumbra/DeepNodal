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
  # public 
  def_name = 'chain'
  def_subobject_name = 'link'
  def_subobject = link

  # protected 
  _inp = None
  _out = None
  _links = None
  _n_links = None
  _unit_link = None

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    # We add links sequentially before declaring as subobjects so
    # the stem version of this function must be overloaded here...

    self.name = name if name is not None else self._def_name
    if self._links is None: return
    for i, _link in enumerate(self._links):
      # no point re-naming links if unit_link true 
      link_name = self.name if self._unit_link else self.name + "/" + self._subobject_name + "_" + str(i)
      _link.set_name(link_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    # We add links sequentially before declaring as subobjects so
    # the stem version of this function must be overloaded here...
    self.dev = dev
    if self._links is None: return
    for _link in self._links:
      _link.set_dev(dev)

#-------------------------------------------------------------------------------
  def add_link(self, creation = None, *args, **kwds):
    # this requires no input since it does not create graph objects
    if self._links is None:
      self._links = []
      self._n_links = 0
      self._unit_link = False
    if creation is None:
      return
    if isinstance(creation, link) and not(len(args)) and not(len(kwds)):
      self._links.append(creation)
    else:
      if 'name' in kwds:
        name = kwds['name']
      elif 'scope' in kwds:
        name = kwds['scope']
      else:
        name = 'link_' + str(self._n_links)
        if self.name is not None:
          name = self.name + "/" + name
      self._links.append(link(name, self.dev))
    self._n_links = len(self._links)
    self._unit_link = self._n_links == 1
    self._links[-1].set_creation(creation, *args, **kwds)
    return self._links[-1]

#-------------------------------------------------------------------------------
  def __call__(self, inp = None):
    self._inp = inp
    self._out = inp
    if self._links is None: return
    for _link in self._links:
      inp = _link.__call__(inp)
    self._out = inp
    return self.ret_out()

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    # Create chain instance if necessary
    if other is None:
      other = chain()
    elif not isinstance(other, chain) and not issubclass(other, chain):
      raise TypeError("Cannot clone chain to class " + str(other))
    elif other._links is not None:
      raise AttributeError("Cannot clone to a chain instance with pre-existing links")

    # All we have to do now is clone the links
    if self._links is not None:
      for _link in self._links:
        other.add_link(_link.clone())

    # Now rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)
    return other

#-------------------------------------------------------------------------------

