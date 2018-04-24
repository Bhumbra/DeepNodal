# Base class for single input and output functionality through a single stream.
#
#
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.concepts.stem import stem
from deepnodal.structures.link import *

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

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(name, dev)

#-------------------------------------------------------------------------------
  def add_link(self, creation = None, *args, **kwds):
    # this requires no input since it does not create graph objects
    if self.links is None:
      self.links = []
      self.n_links = 0
    if creation is None:
      return
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
    self.links[-1].set_creation(creation, *args, **kwds)
    return self.links[-1]

#-------------------------------------------------------------------------------
  def setup(self, inp = None):
    self.inp = inp
    self.out = None
    for _link in self.links:
      inp = _link.setup(inp)
    self.out = inp
    self.set_outputs([{self.name + "/output", self.out}])
    return self.ret_out()

#-------------------------------------------------------------------------------

