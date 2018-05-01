"""
Hypervisor module for Tensorflow. 
"""
# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.supervisor import *
from deepnodal.python.concepts.master import *

#------------------------------------------------------------------------------- 
class hypervisor (supervisor, master, stem):
  """
  A hypervisor is a class that distributes computations for supervised 
  learning across multiple devices.

  It is three things in one class:

  A supervisor: it updates parameters according to gradient-based optimisation.
  A master: it enslaves supervisors to evaluate the gradients on data subbatches.
  A stem: it is effectively the final `trunk' that clones networks as required.

  For syntactic purposes, it is called exactly in the same way as the supervisor
  except at it's instantiation:

  sup = supervisor() and sup = hypervisor() are identical

  sup = supervisor('name', 'dev') and sup = hypervisor('name', 'dev') are identical.

  sup = hypervisor('name', devs = 2) or sup = hypervisor(devs = 2) sets up a 
  hypervisor instance that distributes supervised learning across 2 GPU devices.

  """
  def_name = 'hypervisor'
  unit_device = None        # Boolean flag to denote single device (i.e. supervisor)     
  devs = None               # GPU devices
  n_devs = None             # len(devs)
  clones = None             # Network clones
  slaves = None             # Supervisor slaves

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None, devs = None):
    """
    To instantiate for using 2 GPUs, invoke: 
    
    sup = hypervisor(devs = 2) or sup = hypervisor('name', devs = 2)

    """
    supervisor.__init__(self, name, dev, devs)

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name

    # Set names of supervisors (through master)
    master.set_name(self, self.name)

    # Set names of regimens (through overseer)
    supervisor.set_name(self, self.name)
    
    # Set names of clones (through stem)
    stem.set_name(self, self.name)

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None, devs = None):
    """
    If devs in an integer, it represents the number of GPUs to use.
    """
    # Here dev is either None or a number
    self.dev = dev
    self.devs = devs
    if type(self.devs) is int:
      self.n_devs = self.devs
      self.devs = [Device('gpu', i) for i in range(self.n_devs)]
    if self.devs is not None:
      if self.dev is None:
        self.dev = Device('cpu', 0)
    self.n_devs = 0
    self.unit_dev = self.devs is not None
    if not(self.unit_dev):
      self.n_devs = len(self.devs)
    return self.unit_dev

#-------------------------------------------------------------------------------
  def set_work(self, work = None): 
    # If not(self.unit_device), the slaves and clones are instantiated here
    argout = superviser.set_work(None)
    if self.unit_device: return argout

    # Clone networks
    self.clones = [self.work.clone() for i in range(self.n_devs)]

    # Declare clones as subobjects to stem
    self.set_subobjects(self.clones)

    # Enslave supervisors
    self.slaves = [supervisor() for i in range(self.n_devs)]

    # Declare slaves as subworkers to master
    self.set_subworkers(self.slaves)

    # Rename and redevice clones and slaves
    for i in range(self.n_devs):
      self.clones[i].set_name(self.work.name + "/clone_" + str(i))
      self.clones[i].set_dev(self.devs[i])
      self.slaves[i].set_name(self.name + "/slave_" + str(i))
      self.slaves[i].set_dev(self.devs[i])
      self.slaves[i].set_work(self.clones[i])

#-------------------------------------------------------------------------------
  def set_errorq(self, erq = None, *erq_args, **erq_kwds):
    argout = supervisor.set_errorq(erq, *erq_args, **erq_kwds)
    if self.unit_device: return argout
    return [_slave.set_errorq(erq, *erq_args, **erq_kwds) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def set_costfn(self, cfn = None, *cfn_args, **cfn_kwds):
    argout = supervisor.set_costfn(cfn, *cfn_args, **cfn_kwds)
    if self.unit_device: return argout
    return [_slave.set_costfn(cfn, *cfn_args, **cfn_kwds) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def setup(self, ist = None, gst = None, skip_metrics = False, **kwds):
    if self.unit_device:
      argout = supervisor._setup(ist, gst, skip_metrics, **kwds)
      return argout

    """
    At this point, there is only one correct order to proceed.

    1. Setup the original network - there is no reason for it to be changed even if some
       class member properties are redundant at this stage (e.g. network.outputs)
    2. While setting up the hypervisor as an overseer and trainer, overload _setup_functions
       to include specification parameters (including identity operations for split
       inputs) for slave specifications commanded by the hypervisor (e.g. inputs, optimiser, 
       regimes), and while setting up as a supervisor, the same for the labels.
    3. Before setting up hypersupervisor performance metrics (e.g. cost, loss, gradients),
       complete the setup of all slaves and associated clones.
    4. Setup hypervisor performance metrics and gradients as averages from slaves.
    5. In addition to the delta operations setup parameter update operations for clones.
    """

    # 1. and 2. 
    supervisor._setup(ist, gst, True, **kwds)

    # 3. Complete the setup of all slaves and associated clones.
    [_slave.setup(self.ist, self.gst, skip_metrics, **kwds) for _slave in self.slaves]

    # 4. and 5.
    self._setup_metrics(skip_metrics)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _setup_is_training(self, ist = None): # overloading trainer._setup_is_training(self, ist)
    argout = supervisor._setup_is_training(ist)
    if self.unit_device: return argout
    return [_slave._setup_is_training(self.ist) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def _setup_inputs(self, ist = None): # overloading trainer._setup_inputs()
    argout = supervisor._setup_inputs(ist)
    if self.unit_device: return argout

    # Setup inputs_to_clones through diverging by splitting input batch
    self.inputs_to_clones = [None] * self.n_devs
    for i, _input in enumerate(self.inputs):
      if self.dev is None:
        self.inputs_to_clones[i] = Creation('diverge')(_input, self.n_devs, axis = 0,
                                   name = self.name + '/inputs_to_clones')
      else:
        with Device(self.dev):
          self.inputs_to_clones[i] = Creation('diverge')(_input, self.n_devs, axis = 0,
                                     name = self.name + '/inputs_to_clones')

    # Re-index by clone
    self.inputs_by_clones = [None] * self.n_devs
    for i in range(self.n_devs):
      self.inputs_by_clones[i] = [None] * self.n_subnets
      for j in range(self.n_subnets):
        self.inputs_by_clones[i][j] = self.inputs_to_clones[j][i]

    # Set identity object specifications to the clones
    for _inputs, _clone in zip(self.input_by_clones, self.clones):
      _clone.set_inputs('identity', _inputs) 

    # Now the clones will take care of the rest at the network._setup_inputs() stage
    return argout
    
#-------------------------------------------------------------------------------
  def _setup_learning_rate(self, gst = None): # overloading trainer._setup_learning_rate
    argout = supervisor._setup_learning_rate(gst)
    if self.unit_device: return argout
    for _slave in self.slaves:
      _slave.set_learning_rate(self.lrn, *self.lrn_args, **self.lrn_kwds)
    return argout

#-------------------------------------------------------------------------------
  def _setup_optimiser(self): # overloading trainer._setup_optimiser()
    argout = supervisor._setup_optimiser()
    if self.unit_device: return argout
    for _slave in self.slaves:
      _slave.set_optim(self.opt, *self.opt_args, **self.opt_kwds)
    return argout

#-------------------------------------------------------------------------------
  def set_session(self, session = None): # overloading trainer.set_session(session)
    argout = supervisor.set_session()
    if self.unit_device: return argout
    for _slave in self.slaves:
      _slave.set_session(self.session)
    return argout

#-------------------------------------------------------------------------------
  def _setup_regimes(self, gst = None): # overloading overseer._setup_regimes(gst)
    argout = supervisor._setup_regimes(gst)
    if self.unit_device: return argout
    for _slave in self.slaves:
      _slave.set_regimes(self.regimes)

#-------------------------------------------------------------------------------
  def use_regime(self, using_regime = -1): # overloading overseer.use_regime(using_regime)
    argout = supervisor.use_regimen(using_regime)
    if self.unit_device: return argout
    for _slave in self.slaves:
      _slave.use_regime(self.using_regime)
    return argout

#-------------------------------------------------------------------------------
  def _setup_labels(self): # overloading supervisor._setup_labels()
    argout = supervisor._setup_labels(ist)
    if self.unit_device: return argout

    # Setup labels_to_slaves through diverging by splitting label batch
    if self.dev is None:
      self.labels_to_slaves = Creation('diverge')(self.labels, self.n_devs, axis = 0,
                              name = self.name + '/labels_to_slaves')
    else:
      with Device(self.dev):
        self.labels_to_slaves = Creation('diverge')(self.labels, self.n_devs, axis = 0,
                                name = self.name + '/labels_to_slaves')

    # Set identity object specifications to the slaves
    for _labels, _slave in zip(list(self.labels_to_slaves), self.clones):
      _slave.set_labels('identity', _labels) 

    # Now the slaves will take care of the rest at the supervisor._setup_labels() stage
    return argout
    
#-------------------------------------------------------------------------------

