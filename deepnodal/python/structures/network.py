"""
Network module for Tensorflow. A network is list of inter-connected stacks, 
levels, or streams with accompanying inputs. Network architectures cannot
be specified directly since they do not recognise hierarchical structures.
"""

# Gary Bhumbra
#-------------------------------------------------------------------------------
DEFAULT_INPUT_STRUCTURE_TYPES = ['stream', 'level', 'stack']
DEFAULT_INPUT_INT_DATA_TYPE = 'int32'
DEFAULT_INPUT_FLOAT_DATA_TYPE = 'float32'

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.stem import stem
from deepnodal.python.structures.stack import *

#-------------------------------------------------------------------------------
class network (stem):
  """
  A network comprises of a list of inter-connected subnets. A subnet may be a 
  stack, level, or stream. A network is not an architectural parent to any
  subnet therefore the architecture of subnets cannot be specified through 
  network instances but must be defined separately and fed to a network instance 
  using self.set_subnets([_list_of_subnets_])

  Inputs are specified using network.set_inputs([_list_of_inputs__]) where each
  input may either be a dimension specification, or an instance of a subnet
  whose output is used. The order of inputs much match the order of subnets
  that permit sequential dependencies. Alternatively a single input can be used
  for all subnets as long as the dimension specifications are valid.

  This approach only permits full inputs and outnets of subnets to be 
  corresponded, but a future intent is use dictionaries to permit partial 
  inputs and outnets to be used.

  Using network.set_subnets() will override the names and devices according
  of the subnet components according to settings of the network class instance.
  """

  # public
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

  # protected
  _inp = None                  # Graph of inputs
  _inputs = None               # list of input dictionaries in form {name: input}

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    stem.__init__(self, name, dev)
    self.set_subnets() # defaults to nothing

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    # We collate subnets declaring them as subobjects so
    # the stem version of this function must be overloaded here...

    self.name = name if name is not None else self.def_name
    if self.subnets is None: return
    for i, _subnet in enumerate(self.subnets):
      # no point re-naming subnets if unit_subnet is true 
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
      """
      if isinstance(subnet, stream) or issubclass(subnet, stream):
        self.type_subnets[i] = 'stream'
      elif isinstance(subnet, level) or issubclass(subnet, level):
        self.type_subnets[i] = 'level'
      elif isinstance(subnet, stack) or issubclass(subnet, stack):
        self.type_subnets[i] = 'stack'
      else:
        raise TypeError("Unknown subnet type: " + str(subnet))
      """
      if isinstance(subnet, stream):
        self.type_subnets[i] = 'stream'
      elif isinstance(subnet, level):
        self.type_subnets[i] = 'level'
      elif isinstance(subnet, stack):
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
  #def set_inputs(self, *inputs, inputs_dtype = None):
  def set_inputs(self, inputs = None, *inputs_args, **inputs_kwds):
    self.inputs = inputs
    self.inputs_args = inputs_args
    self.inputs_kwds = dict(inputs_kwds)
    self.type_inputs = None
    if self.inputs is None: return
    if type(self.inputs) is int:
      self.inputs = [[self.inputs]] * self.n_subnets
    elif type(self.inputs) is not list:
      self.inputs = [self.inputs] * self.n_subnets
    elif len(self.inputs):
      if type(self.inputs[0]) is int: # in case of a single input specification
        self.inputs = [self.inputs]
    if len(self.inputs) != self.n_subnets:
      raise ValueError("Number of inputs must match number of subnets")
    self.type_inputs = ['unspecified'] * self.n_subnets
    for i, inp in enumerate(self.inputs):
      if type(inp) is list or type(inp) is tuple or type(inp) is int or type(inp) is set or type(inp) is dict:
        self.type_inputs[i] = 'arch'
      elif type(inp) is str:
        self.type_inputs[i] = str(inp)
      elif isinstance(inp, stream) or issubclass(inp, stream):
        self.type_inputs[i] = 'stream'
      elif isinstance(inp, level) or issubclass(inp, level):
        self.type_inputs[i] = 'level'
      elif isinstance(inp, stack) or issubclass(inp, stack):
        self.type_inputs[i] = 'stack'
      elif callable(inp):
        self.type_inputs[i] = 'callable'
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
    if other is None:
      other = network()
    elif not isinstance(other, network) and issubclass(other, network):
      raise TypeError("Cannot clone network to class " + str(other))
    elif other.subnets is not None:
      raise AttributeError("Cannot clone to a network instance with pre-existing subnets")

    # Clone the subnets one-by-one and declare as subnets of new network
    other.set_subnets([_subnet.clone() for _subnet in self.subnets])

    # Raw architectural inputs will be copied over whereas inputs that are subnets will be matched clone-for-clone
    other_inputs = list(self.inputs)

    for i, inp in enumerate(other_inputs):
      if self.type_inputs[i] in DEFAULT_INPUT_STRUCTURE_TYPES:
        index = None
        for j, obj in enumerate(self.subnets):
          if inp == obj:
            index = j
        if index is None:
          raise AttributeError("Cannot clone unidentifiable non-architectural input: "+str(inp))
        else:
          other_inputs[i] = other.subnets[index]
      elif self.type_inputs[i] != "arch": # i.e. 'unspecified'
        raise ValueError("Cannot clone functional forms of unspecified inputs")
    
    other.set_inputs(other_inputs)

    # Rename and redevice
    other.set_name(self.name)
    other.set_dev(self.dev)

    return other

#-------------------------------------------------------------------------------
  def __call__(self, ist = None, _called = True, **kwds):

    if self.inputs is None: return

    # Unify ist across all subnets
    self.set_is_training(ist)

    # Call inputs
    self._call_inputs()

    # Call subnets
    self._call_subnets()
    self.set_called(_called)

    # Collate architectural parameters
    self._setup_params()

    # Collate normalisation moments
    self._setup_moments()

    # Collate regularisation parameters
    self._setup_reguln()

    # Collate updates
    self._setup_updates()

    # Collate list of outputs
    self._setup_outputs()

    return self.ret_out()

#-------------------------------------------------------------------------------
  def _call_inputs(self):
    self._inp = [None] * self.n_subnets
    self._inp_args = [None] * self.n_subnets
    self._inp_kwds = [None] * self.n_subnets
    self._inputs = [None] * self.n_subnets
    for i in range(self.n_subnets):
      if self.type_inputs[i] in DEFAULT_INPUT_STRUCTURE_TYPES:
        self._inp[i] = self.inputs[i].ret_out()
        if self._inp[i] is None:
          raise ValueError("Sequential input logic violated.")
        self._inputs[i] = mapping({self._inp[i].name: self._inp[i]})
      elif self.type_inputs[i] == 'arch':
        kwds = dict(self.inputs_kwds)
        if 'dtype' not in kwds:
          if type(self.inputs[i]) is set or type(self.inputs[i]) is dict:
            kwds.update({'dtype': Dtype(DEFAULT_INPUT_INT_DATA_TYPE)})
          else:
            kwds.update({'dtype': Dtype(DEFAULT_INPUT_FLOAT_DATA_TYPE)})
        inp_dim = [None]
        if type(self.inputs[i]) is set:
          if len(self.inputs[i]) > 1:
            raise TypeError("Input set specification must contain no more than 1 element")
        for dim in self.inputs[i]:
          inp_dim.append(dim)
        self._inp_args[i] = (kwds['dtype']),
        self._inp_kwds[i] = {'shape':list(inp_dim)}
        inp_name = self.name + "/inputs_" + str(i)
        if self.dev is None:
          self._inp[i] = Creation('tensor')(*self._inp_args[i], name = inp_name, **self._inp_kwds[i])        
        else:
          with Device(self.dev):
            self._inp[i] = Creation('tensor')(*self._inp_args[i], name = inp_name, **self._inp_kwds[i])        
        self._inputs[i] = mapping({inp_name: self._inp[i]})
      elif callable(Creation(self.inputs[i])): # the inputs creation is being over-ridden externally
        inp_name = self.name + "/inputs_" + str(i),
        if self.dev is None:
          self._inp[i] = Creation(self.inputs[i])(self.inputs_args[i],
                                                  name = self.name + "/inputs_" + str(i),
                                                  **self.inputs_kwds)
        else:
          with Device(self.dev):
            self._inp[i] = Creation(self.inputs[i])(self.inputs_args[i],
                                                    name = self.name + "/inputs_" + str(i),
                                                    **self.inputs_kwds)
        self._inputs[i] = mapping({inp_name: self._inp[i]})
      else:
        self._inp[i] = self.inputs[i]
        self._inputs[i] = mapping({self._inp[i].name: self._inp[i]})

#-------------------------------------------------------------------------------
  def _call_subnets(self):
    for i in range(self.n_subnets):
      self.subnets[i].__call__(self._inp[i])
    self._out = [subnet.ret_out() for subnet in self.subnets]

    # Declare the subnets as subobjects to have access to their respective parameters
    self.set_subobjects(self.subnets) 

    return self.ret_out()

#-------------------------------------------------------------------------------
  def ret_inputs(self, input_spec = None):
    if not(self.ret_called()): return self, network.ret_inputs, input_spec
    if input_spec is None:
      return self._inputs
    inputs = []
    if type(input_spec) is bool:
      if input_spec:
        inputs = self._inputs
      return inputs
    elif type(input_spec) is int: 
      input_spec = [input_spec]
      return [self._inputs[input_spec]]
    input_spec = np.array(input_spec)
    if input_spec.dtype is np.dtype('int'):
      inputs = [self._inputs[_input_spec] for _input_spec in input_spec]
      return inputs
    for i, _input_spec in enumerate(input_spec):
      if _input_spec:
        inputs.append(self._inputs[i])
    return inputs

#-------------------------------------------------------------------------------
  def _ret_input(self, input_spec = None):
    if not(self.ret_called()): return self, network._ret_input, input_spec
    inputs = self.ret_inputs(input_spec)
    if len(inputs) != 1:
      raise ValueError("Specification " + str(input_spec) + 
                       " returns " + str(len(inputs)) + " results.")

    return inputs[0]

#-------------------------------------------------------------------------------
  def ret_input(self, input_spec = None):
    if not(self.ret_called()): return self, network.ret_input, input_spec
    inputs = self._ret_input(input_spec)
    return list(inputs.values())[0]

#-------------------------------------------------------------------------------
  def _setup_reguln(self):
    self._reguln = {'loss': [], 'grad': [], 'vars': []}
    keys = list(self._reguln.keys())
    for subobject in self._subobjects:
      sub_reg = subobject.ret_reguln()
      for key in keys:
        self._reguln[key].extend(sub_reg[key])
    return self._reguln

#-------------------------------------------------------------------------------
  def ret_reguln(self):
    return self._setup_reguln()

#-------------------------------------------------------------------------------
