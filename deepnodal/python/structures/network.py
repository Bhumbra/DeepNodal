"""
Network module for Tensorflow. A network is list of inter-connected stacks, 
levels, or streams with accompanying inputs. Network archectures must not
be specified directly since they do not recognise hierarchical structures.
"""

# Gary Bhumbra
#-------------------------------------------------------------------------------
DEFAULT_INPUT_DATA_TYPE = 'float32'

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import stem
from deepnodal.python.structures.stack import *

#-------------------------------------------------------------------------------
class network (stem):
  """
  A network comprises of a list of inter-connected subnets. A subnet may be a 
  stack, level, or stream. The architecture of subnets cannot be specified 
  through network instances but must be defined separately and fed
  to a network instance using self.set_subnets([__list_of_subnets__])

  Inputs are specified using network.set_inputs([_list_of_inputs__]) where each
  input may either be a dimension specification, or an instance of a subnet
  whose output is used. The order of inputs much match the order of subnets
  that permit sequential dependencies. Alternatively a single input can be used
  for all subnets as long as the dimension specifications are valid.

  This approach only permits full inputs and outnets of subnets to be 
  corresponded, but a future intent is use dictionaries to permit partial 
  inputs and outnets to be used.
  """

  def_name = 'network'
  subnets = None               # Subnet instances excluding inputs
  type_subnets = None          # 'stack', 'level', or 'stream'
  n_subnets = None             # len(subnets)
  unit_subnet = None           # n_subnet == 1
  outnets = None               # Those subnets with outputs to other subnets
  n_outnets = None             # len(outnets)
  unit_outnet = None           # n_outnet == 1
  inputs = None                # Inputs ordered to matching subnets
  type_inputs = None           # 'stack', 'level', 'stream', or 'arch'

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)
    self.set_subnets() # defaults to nothing
    self.setup()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    # We collate subnets declaring them as subobjects so
    # the stem version of this function must be overloaded here...

    self.name = name if name is not None else self.def_name
    if self.subnets is None: return
    for i, _subnet in enumerate(self.subnets):
      # no point re-naming subnets if unit_subnet true 
      subnet_name = self.name if self.unit_subnet else self.name + "/subnet" + "_" + str(i)
      _subnet.set_name(subnet_name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    # We collate subnets declaring them as subobjects so
    # the stem version of this function must be overloaded here...
    self.dev = dev
    if self.subnets is None: return
    for _subnet in self.subnets:
      _subnet.set_dev(dev)

#-------------------------------------------------------------------------------
  def set_subnets(self, subnets = None):
    self.subnets = subnets
    self.type_subnets = None
    if self.subnets is None: return self.type_subnets
    if type(self.subnets) is not list:
      self.subnets = [self.subnets]
    self.n_subnets = len(self.subnets)
    self.unit_subnet = self.n_subnets == 1
    self.type_subnets = [None] * self.n_subnets
    for i, subnet in enumerate(self.subnets):
      if isinstance(subnet, stream) or issubclass(subnet, stream):
        self.type_subnets[i] = 'stream'
      elif isinstance(subnet, level) or issubclass(subnet, level):
        self.type_subnets[i] = 'level'
      elif isinstance(subnet, stack) or issubclass(subnet, stack):
        self.type_subnets[i] = 'stack'
      else:
        raise TypeError("Unknown subnet type: " + str(subnet))
    self.set_outnets()
    self.set_name(self.name) # this renames the subnets
    self.set_dev(self.dev)   # this redevices the subnets
    return self.type_subnets
    
#-------------------------------------------------------------------------------
  def set_outnets(self, indices = None): # for now we will only have one output
    self.outnets = None
    self.n_outnets = 0
    if self.subnets is None: return self.outnets
    if indices is None: 
      indices = list(range(self.n_subnets))
    elif type(indices) is int: 
      indices = [indices]
    self.outnets = []
    for i, subnet in enumerate(self.subnets):
      if i in indices:
        self.outnets.append(subnet)
    self.n_outnets = len(self.outnets)
    self.unit_outnet = self.n_outnets == 1
    return self.outnets

#-------------------------------------------------------------------------------
  def set_inputs(self, inputs = None):
    self.inputs = inputs
    self.type_inputs = None
    if self.inputs is None: return
    if type(self.inputs) is not list:
      self.inputs = [self.inputs] * self.n_subnets
    elif len(self.inputs):
      if type(self.inputs[0]) is int: # in case of a single input specification
        self.inputs = [self.inputs]
    if len(self.inputs) != self.n_subnets:
      raise ValueError("Number of inputs must match number of subnets")
    self.type_inputs = ['arch'] * self.n_subnets
    for i, inp in enumerate(self.inputs):
      if isinstance(inp, stream) or issubclass(inp, stream):
        self.type_inputs[i] = 'stream'
      elif isinstance(inp, level) or issubclass(inp, level):
        self.type_inputs[i] = 'level'
      elif isinstance(inp, stack) or issubclass(inp, stack):
        self.type_inputs[i] = 'stack'
    return self.type_inputs

#-------------------------------------------------------------------------------
  def set_is_training(self, ist = None):
    self.ist = ist
    if self.subnets is None: return
    for subnet in self.subnets:
      subnet.set_is_training(ist)

#-------------------------------------------------------------------------------
  def set_dropout(self, spec = None, *args, **kwds):
    """
    We support this specifically to allow updates mid-session.
    """
    if type(spec) is not list: spec = [spec] * self.n_subnets
    if type(args) is not list: args = [args] * self.n_subnets
    if type(kwds) is not list: kwds = [kwds] * self.n_subnets
    return [subnet.set_dropout(spec[i], *args[i], **kwds[i]) for i, subnet in enumerate(self.subnets)]

#-------------------------------------------------------------------------------
  def clone(self, other = None):
    raise TypeError("While subnets can be cloned, networks cannot for safety.")

#-------------------------------------------------------------------------------
  def setup(self, ist = None, **kwds):
    if self.inputs is None:
      return
    self.set_is_training(ist)
    self.inp = [None] * self.n_subnets
    for i in range(self.n_subnets):
      if self.type_inputs[i] != 'arch':
        self.inp[i] = self.inputs[i].ret_out()
        if self.inp[i] is None:
          raise ValueError("Sequential input logic violated.")
      else:
        kwargs = dict(kwds)
        if 'dtype' not in kwargs:
          kwargs.update({'dtype':Dtype(DEFAULT_INPUT_DATA_TYPE)})
        inp_dim = [None]
        for dim in self.inputs[i]:
          inp_dim.append(dim)
        if self.dev is None:
          self.inp[i] = Creation('tensor')(kwargs['dtype'], shape = inp_dim,
                                       name = self.name + "/inputs_" + str(i))
        else:
          with Device(self.dev):
            self.inp[i] = Creation('tensor')(kwargs['dtype'], shape = inp_dim,
                                         name = self.name + "/inputs_" + str(i))
      self.subnets[i].setup(self.inp[i])
    self.out = [subnet.ret_out() for subnet in self.subnets]

    # Declare the subnets as subobjects to have access to their respective parameters
    self.set_subobjects(self.subnets) 

    # Collate architectural parameters
    self.setup_params()

    # Collate list of outputs
    self.setup_outputs()

    return self.ret_out()

#-------------------------------------------------------------------------------
