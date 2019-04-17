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
  unit_dev = None        # Boolean flag to denote single device (i.e. supervisor)     
  devs = None               # GPU devices
  n_devs = None             # len(devs)
  clones = None             # Network clones
  slaves = None             # Supervisor slaves
  param_ops = None          # Parameter operations for the master to update slaves.   
  moments = None            # Moment mappings
  n_moments = None          # len(Moment)

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None, devs = None):
    """
    To instantiate for using 2 GPUs, invoke: 
    
    sup = hypervisor(devs = 2) or sup = hypervisor('name', devs = 2)

    """
    supervisor.__init__(self)
    master.__init__(self)
    stem.__init__(self)
    self.set_name(name)
    self.set_dev(dev, devs)

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name

    # Set names of supervisors (through master)
    master.set_name(self, self.name)

    # Set names of schedules (through overseer)
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
    self.unit_dev = self.devs is None
    if not(self.unit_dev):
      self.n_devs = len(self.devs)
    return self.unit_dev

#-------------------------------------------------------------------------------
  def set_work(self, work = None): 
    # If not(self.unit_dev), the slaves and clones are instantiated here
    argout = supervisor.set_work(self, work)
    if self.unit_dev: return argout

    # Clone networks
    self.clones = [self.work.clone() for i in range(self.n_devs)]

    # Remove weight initialisers from clones (GPUs are less flexible)
    for clone in self.clones:
      for subnet in clone.subnets:
        subnet.set_weights(None)

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
    argout = supervisor.set_errorq(self, erq, *erq_args, **erq_kwds)
    if self.unit_dev: return argout
    return [_slave.set_errorq(erq, *erq_args, **erq_kwds) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def set_costfn(self, cfn = None, *cfn_args, **cfn_kwds):
    argout = supervisor.set_costfn(self, cfn, *cfn_args, **cfn_kwds)
    if self.unit_dev: return argout
    return [_slave.set_costfn(cfn, *cfn_args, **cfn_kwds) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def __call__(self, ist = None, gst = None, skip_metrics = False, _called = True, **kwds):
    if self.work is None: return
    if self.unit_dev:
      argout = supervisor.__call__(self, ist, gst, skip_metrics, _called, **kwds)
      return argout

    """
    At this point, there is only one correct order to proceed.

    1. Setup the original network - there is no reason for it to be changed even if some
       class member properties are redundant at this stage (e.g. network.outputs)
    2. While setting up the hypervisor as an overseer and trainer, call overloaded 
       __call__ functions to include specification parameters (including identity 
       operations for split inputs) for slave specifications commanded by the hypervisor 
       (e.g. inputs, optimiser, schedules), and while setting up as a supervisor, the same 
       for the labels.
    3. Re-assign the original network outputs with concatenated outputs from those slaves
    4. Before setting up hypersupervisor performance metrics (e.g. cost, loss, gradients),
       complete the setup of all slaves and associated clones.
    5. Setup hypervisor performance metrics and gradients as averages from slaves.
    6. In addition to the apply operations setup parameter update operations for clones.
    """

    # 1. and 2. 
    supervisor.__call__(self, ist, gst, True, _called, **kwds)

    # 3. and 4.
    [_slave.__call__(self.ist, self.gst, skip_metrics, **kwds) for _slave in self.slaves]
    self._call_outputs(True)

    # 5. and 6.
    self._call_metrics(skip_metrics)

    self.set_called(_called)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _call_is_training(self, ist = None): # overloading trainer._call_is_training(self, ist)
    argout = supervisor._call_is_training(self, ist)
    if self.unit_dev: return argout
    return [_slave._call_is_training(self.ist) for _slave in self.slaves]

#-------------------------------------------------------------------------------
  def _call_inputs(self): # overloading trainer._call_inputs()
    argout = supervisor._call_inputs(self)
    if self.unit_dev: return argout

    # Setup inputs_to_clones through diverging by splitting input batch
    self.inputs_to_clones = [None] * len(self.inputs)
    for i, _input in enumerate(self.inputs):
      if self.dev is None:
        self.inputs_to_clones[i] = Creation('diverge')(_input, self.n_devs, axis = 0,
                                   name = self.name + '/batch/inputs_to_clones')
      else:
        with Device(self.dev):
          self.inputs_to_clones[i] = Creation('diverge')(_input, self.n_devs, axis = 0,
                                     name = self.name + '/batch/inputs_to_clones')

    # Re-index by clone
    self.inputs_by_clones = [None] * self.n_devs
    for i in range(self.n_devs):
      self.inputs_by_clones[i] = [None] * len(self.inputs)
      for j in range(len(self.inputs)):
        self.inputs_by_clones[i][j] = self.inputs_to_clones[j][i]

    # Set identity object specifications to the clones
    for _inputs, _clone in zip(self.inputs_by_clones, self.clones):
      _clone.set_inputs(_inputs) 

    # Now the clones will take care of the rest at the network._call_inputs() stage
    return argout
    
#-------------------------------------------------------------------------------
  def _call_outputs(self, reassign = False): # overloading trainer._call_outputs()
    argout = supervisor._call_outputs(self)
    if self.unit_dev: return argout
    if not(reassign): return argout
    slave_outputs = [_slave.outputs for _slave in self.slaves]

    slave_outputs_by_slave = [None] * self.n_outputs
    for i in range(self.n_outputs):
      slave_outputs_by_slave[i] = [None] * self.n_devs
      for j in range(self.n_devs):
        slave_outputs_by_slave[i][j] = slave_outputs[j][i]

    for i in range(self.n_outputs):
      if self.dev is None:
        self.outputs[i] = Creation('con')(slave_outputs_by_slave[i], axis=0,
                          name = self.name + "/" + self.work.name + "/update_ops/" + self.output_names[i])
      else:
        with Device(self.dev):
          self.outputs[i] = Creation('con')(slave_outputs_by_slave[i], axis=0,
                            name = self.name + "/" + self.work.name + "/update_ops/" + self.output_names[i])
    return self.outputs

#-------------------------------------------------------------------------------
  def _call_learning_rate(self, gst = None): # overloading trainer._call_learning_rate
    argout = supervisor._call_learning_rate(self, gst)
    if self.unit_dev: return argout
    for _slave in self.slaves:
      _slave.set_learning_rate(self._lrn, *self._lrn_args, **self._lrn_kwds)
    return argout

#-------------------------------------------------------------------------------
  def _call_optimiser(self): # overloading trainer._call_optimiser()
    argout = supervisor._call_optimiser(self)
    if self.unit_dev: return argout
    for _slave in self.slaves:
      _slave.set_optimiser(self._opt, *self._opt_args, **self._opt_kwds)
    return argout

#-------------------------------------------------------------------------------
  def set_session(self, session = None): # overloading trainer.set_session(session)
    argout = supervisor.set_session(self, session)
    if self.unit_dev: return argout
    for _slave in self.slaves:
      _slave.set_session(self.session)
    return argout

#-------------------------------------------------------------------------------
  def _setup_schedules(self, gst = None): # overloading overseer._setup_schedules(gst)
    argout = supervisor._setup_schedules(self, gst)
    if self.unit_dev: return argout
    for _slave in self.slaves:
      _slave.set_schedules(self.schedules)

#-------------------------------------------------------------------------------
  def use_schedule(self, using_schedule = -1): # overloading overseer.use_schedule(using_schedule)
    argout = supervisor.use_schedule(self, using_schedule)
    if self.unit_dev: return argout
    for _slave in self.slaves:
      _slave.use_schedule(self.using_schedule)
    return argout

#-------------------------------------------------------------------------------
  def _call_labels(self): # overloading supervisor._call_labels()
    argout = supervisor._call_labels(self)
    if self.unit_dev: return argout

    # Setup labels_to_slaves through diverging by splitting label batch
    if self.dev is None:
      self.labels_to_slaves = Creation('diverge')(self.labels, self.n_devs, axis = 0,
                              name = self.name + '/batch/labels_to_slaves')
    else:
      with Device(self.dev):
        self.labels_to_slaves = Creation('diverge')(self.labels, self.n_devs, axis = 0,
                                name = self.name + '/batch/labels_to_slaves')

    # Set identity object specifications to the slaves
    for _labels, _slave in zip(list(self.labels_to_slaves), self.slaves):
      _slave.set_labels(_labels) 

    # Now the slaves will take care of the rest at the supervisor._setup_labels() stage
    return argout
    
#-------------------------------------------------------------------------------
  def _call_errors(self): # overloading supervisor._call_errors()
    if self.unit_dev:
      return supervisor._call_errors(self)

    # If not(self.unit_dev), hypervisor doesn't care about its own hat values
    # because it never evaluates them.

    slave_errors = [_slave.errors for _slave in self.slaves]
    if type(slave_errors) is not list:
      if self.dev is None:
        self.errors = Creation('mean')(Creation('pack')(slave_errors,
                      name = self.name + "/metrics/error_quotients"),
                      name = self.name + "/metrics/error_quotient")
      else:
        with Device(self.dev):
          self.errors = Creation('mean')(Creation('pack')(slave_errors,
                        name = self.name + "/metrics/error_quotients"),
                        name = self.name + "/metrics/error_quotient")
      K = ['']

    else:
      slave_errors_by_slave = [None] * len(self.erq_args[0])
      for i in range(len(self.erq_args[0])):
        slave_errors_by_slave[i] = [None] * self.n_devs
        for j in range(self.n_devs):
          slave_errors_by_slave[i][j] = slave_errors[j][i]

      self.errors = [None] * len(self.erq_args[0])
      K = [None] * len(self.erq_args[0])
      for i, k in enumerate(self.erq_args[0]):
        if self.dev is None:
          self.errors[i] = Creation('mean')(Creation('pack')(slave_errors_by_slave[i],
                           name = self.name + "/metrics/error_quotients_" + str(k)), 
                           name = self.name + "/metrics/error_quotient_" + str(k))
        else:
          with Device(self.dev):
            self.errors[i] = Creation('mean')(Creation('pack')(slave_errors_by_slave[i],
                             name = self.name + "/metrics/error_quotients_" + str(k)), 
                             name = self.name + "/metrics/error_quotient_" + str(k))
        K[i] = '_{}'.format(k)

    self.error_metrics = [None] * len(self.errors)
    for i in range(len(self.errors)):
      self.error_metrics[i] = self.add_metric()
      self.error_metrics[i].set_label('ERROR' + K[i], 'train', 'tests')
      self.error_metrics[i].__call__(self.errors[i])
    return self.errors

#-------------------------------------------------------------------------------
  def _call_costfn(self): # overloading supervisor._call_costfn()
    if self.unit_dev:
      return supervisor._call_costfn(self)
      
    slave_costs = [_slave.cost for _slave in self.slaves]
    if self.dev is None:
      self.cost = Creation('mean')(Creation('pack')(slave_costs,
                  name = self.name + "/metrics/costs"),
                  name = self.name + "/metrics/cost")
    else:
      with Device(self.dev):
        self.cost = Creation('mean')(Creation('pack')(slave_costs,
                    name = self.name + "/metrics/costs"),
                    name = self.name + "/metrics/cost")
    self.cost_metric = self.add_metric()
    self.cost_metric.set_label('COST', 'train', 'tests')
    self.cost_metric.__call__(self.cost)
    return self.cost

#-------------------------------------------------------------------------------
  def _call_losses(self): # overloading supervisor._call_losses()
    if self.unit_dev:
      return supervisor._call_losses(self)
      
    slave_losses = [_slave.loss for _slave in self.slaves]
    if self.dev is None:
      self.loss = Creation('mean')(Creation('pack')(slave_losses,
                  name = self.name + "/metrics/losses"),
                  name = self.name + "/metrics/loss")
    else:
      with Device(self.dev):
        self.loss = Creation('mean')(Creation('pack')(slave_losses,
                    name = self.name + "/metrics/losses"),
                    name = self.name + "/metrics/loss")
    self.loss_metric = self.add_metric()
    self.loss_metric.set_label('LOSS', 'train', 'tests')
    self.loss_metric.__call__(self.loss)
    return self.loss

#-------------------------------------------------------------------------------
  def _call_slave_means(self, values_by_slave, value_names):
    n_values = len(value_names)
    slave_values = [None] * n_values
    for j in range(n_values):
      slave_values[j] = [None] * self.n_devs
      for i in range(self.n_devs):
        value_name = self.name + "/slave_" + str(i) + "/" + value_names[i]
        if self.slaves[i].dev is None:
          slave_values[j][i] = Creation('aug_dims')(values_by_slave[i][j], 0,
                                  name=value_name)
        else:
          with Device(self.slaves[i].dev):
            slave_values[j][i] = Creation('aug_dims')(values_by_slave[i][j], 0,
                                    name=value_name)
    
    mean_values = [None] * n_values
    for i in range(n_values):
      value_name = self.name + "/batch/" + value_names[i]
      if self.dev is None:
        mean_values[i] = Creation('mean')(Creation('con')(slave_values[i], axis=0,
                         name=value_name + "_con"), axis=0,
                         name=value_name + "_mean")
      else:
        with Device(self.dev):
          mean_values[i] = Creation('mean')(Creation('con')(slave_values[i], axis=0,
                           name=value_name + "_con"), axis=0,
                           name=value_name + "_mean")
    return slave_values, mean_values

#-------------------------------------------------------------------------------
  def _call_gradients(self): # overloading supervisor._call_gradients()
    argout = supervisor._call_gradients(self, skip_reg_grad=not self.unit_dev)
    if self.unit_dev: return argout
    # Note the slaves gradients will already include any regularisation deltas
    slave_gradients = [_slave.gradients for _slave in self.slaves]
    self.slave_grad, self.gradients = self._call_slave_means(
                                        slave_gradients, self.gradient_names)

    for i, grad in enumerate(self.gradients):
      self.grad_and_vars[i] = list(self.grad_and_vars[i])
      self.grad_and_vars[i][0] = grad
      self.grad_and_vars[i] = tuple(self.grad_and_vars[i])

    return self.gradients

#-------------------------------------------------------------------------------
  def _call_preta_ops(self): # overloading supervisor._call_preta_ops()
    """
    For multidevice superivsed learning, this comprises of three stages:
    1. Assign clones with parameters (weights & biases) of original model.
    2. Calculate loss gradients of master (and therefore all slaves)
    3. Apply gradient updates to master parameters

    - only step 1 needs particularly special treatment, extending preta_ops
    """
    if self.unit_dev: 
      return supervisor._call_preta_ops(self)

    self._call_param_ops()
    self._call_moment_ops()

    # Parameter updates are schedule-dependent
    self.lrate_ops = [None] * self.n_schedules # learning rate ops
    self.preta_ops = [None] * self.n_schedules # pre-training ops
    for i in range(self.n_schedules):
      with variable_scope(self.name + "/schedules/schedule_"+str(i), reuse=Flag('auto_reuse')):
        self.lrate_ops[i] = self.learning_rate.assign(self.schedules[i].learning_rate)
        self.preta_ops[i] = Creation('combine')(self.lrate_ops[i], self.batch_size_op, 
                                                self.param_ops, self.moment_preta_ops)
    return self.preta_ops

#-------------------------------------------------------------------------------
  def _call_param_ops(self):
    # Collate operations that assign master parameters
    self.param_ops = [None] * self.n_devs * self.n_params
    k = -1
    for _slave in self.slaves:
      for i in range(self.n_params):
        k += 1
        var_scope = self.name + '/assign_to_slaves/' + self.variable_names[i]
        with Scope('var', var_scope, Flag('auto_reuse')):
          if self.dev is None:
            self.param_ops[k] = _slave.variables[i].assign(self.variables[i])
          else:
            with Device(self.dev):
              self.param_ops[k] = _slave.variables[i].assign(self.variables[i])
    return self.param_ops

#-------------------------------------------------------------------------------
  def _call_moment_ops(self): # overloading supervisor._call_preta_ops()
    # Call moment means and collate operations that assign master moments
    slave_moments = [None] * len(self.slaves)
    for i, slave in enumerate(self.slaves):
      slave_moments[i] = [list(moment_dict.values())[0] \
                          for moment_dict in slave.moments]
    self.slave_moments, self.mean_moments = self._call_slave_means(
                                            slave_moments, self.moment_names)
    self.moment_preta_ops = []
    self.moment_posta_ops = []
    if not self.n_moments:
      return self.moment_preta_ops, self.moment_posta_ops
    self.moment_preta_ops = [None] * self.n_devs * self.n_moments
    self.moment_posta_ops = [None] * self.n_devs * self.n_moments
    k = -1
    for _slave in self.slaves:
      for i in range(self.n_moments):
        k += 1
        master_object = list(self.moments[i].values())[0]
        slave_object = list(_slave.moments[i].values())[0]
        mean_object = self.mean_moments[i]
        var_scope_to = self.name + '/assign_to_slaves/' + self.moment_names[i]
        var_scope_from = self.name + '/assign_from_slaves/' + self.moment_names[i]
        with Scope('var', var_scope_to, Flag('auto_reuse')):
          if self.dev is None:
            self.moment_preta_ops[k] = slave_object.assign(master_object)
          else:
            with Device(self.dev):
              self.moment_preta_ops[k] = slave_object.assign(master_object)
        with Scope('var', var_scope_from, Flag('auto_reuse')):
          if self.dev is None:
            self.moment_posta_ops[k] = master_object.assign(mean_object)
          else:
            with Device(self.dev):
              self.moment_posta_ops[k] = master_object.assign(mean_object)

    return self.moment_preta_ops, self.moment_posta_ops

#-------------------------------------------------------------------------------
  def _call_posta_ops(self): # overloading supervisor._call_post_ops
    argout = supervisor._call_posta_ops(self)
    if self.unit_dev: return argout

    # Call moment averages
    self.posta_ops = self.moment_preta_ops

    # Combine post-training ops with apply-ops
    self.train_ops = [None] * self.n_schedules
    for i in range(self.n_schedules):
      if self.dev is None:
        with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/train", reuse=Flag('auto_reuse')):
          self.train_ops[i] = Creation('combine')(self.apply_ops[i], self.posta_ops)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/train", reuse=Flag('auto_reuse')):
            self.train_ops[i] = Creation('combine')(self.apply_ops[i], self.posta_ops)
    return self.train_ops
#-------------------------------------------------------------------------------
  def use_schedule(self, using_schedule = -1, _update_dropout=True): 
    """ overloads overseer.use_schedules """
    if self.unit_dev: 
      return supervisor.use_schedule(self, using_schedule, _update_dropout)
    update_schedule = supervisor.use_schedule(self, using_schedule, False)
    if not update_schedule or not _update_dropout : return update_schedule
    for slave in self.slaves:
      slave.work.set_dropout(self.session, self.schedules[self.using_schedule].dro)
    return update_schedule

#-------------------------------------------------------------------------------
  def test(self, *args, **kwds): # overloading supervisor.test(*args)
    if self.param_ops is not None: # all we need to do update the slave parameters
      self.session.run(self.param_ops, feed_dict = {})
    return supervisor.test(self, *args, **kwds)

#-------------------------------------------------------------------------------
