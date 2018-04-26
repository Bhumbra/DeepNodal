"""
Supervisor module for Tensorflow. 
"""

DEFAULT_OPTIMISER = 'sgd'
DEFAULT_COST_FUNCTION = 'mce'
DEFAULT_LABEL = 'tensor'
DEFAULT_LABEL_DTYPE = 'int64'
DEFAULT_ERROR_QUOTIENT = 'in_top_k_error'
DEFAULT_LEARNING_RATE = 0.01

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.trainer import *

#------------------------------------------------------------------------------- 
class supervisor (trainer):
  """
  A supervisor is a trainer receiving an expected output data set (called labels) 
  for each input  data set that uses a gradient-based optimiser (e.g. stochastic
  gradient descent) to update the parameters within the trainee architecture 
  according to pre-specified learning regimens based on a cost function. 
  
  The convention adopted here is as follows:

  Loss = Cost + regularisation losses

  """
  def_name = 'supervisor'
  cfn = None                   # cost function
  lbl = None                   # labels
  erq = None                   # error quotient 
  gradients = None             # gradients
  grad_and_vars = None         # gradients and variables
  regimen_grad_and_vars = None # grad_and_vars relevant to each regimen
  hat_values = None            # object to compare labels for error calculations
  arch_out = None              # object to compare labels for cost calculations 

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    trainer.__init__(self, name, dev)
    self.set_optimiser()
    self.set_labels()
    self.set_errorq()
    self.set_costfn()

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = DEFAULT_OPTIMISER, *opt_args, **opt_kwds):
    """
    opt = 'sgd' or 'adam' etc...
    """
    trainer.set_optimiser(self, opt, *opt_args, **opt_kwds)

#-------------------------------------------------------------------------------
  def set_labels(self, lbl = None, *lbl_args, **lbl_kwds):
    """
    lbl = 'var'
    lbl_kwds = {'dtype': 'int64'}
    """
    self.lbl = lbl
    self.lbl_args = lbl_args
    self.lbl_kwds = dict(lbl_kwds)
    if self.lbl is None: self.lbl = DEFAULT_LABEL
    if 'dtype' not in self.lbl_kwds:
      self.lbl_kwds.update({'dtype': Dtype(DEFAULT_LABEL_DTYPE)})
    if 'name' not in self.lbl_kwds:
      self.lbl_kwds.update({'name': self.name + "/labels"})
    
#-------------------------------------------------------------------------------
  def set_errorq(self, erq = None, *erq_args, **erq_kwds):
    """

    erq = 'mse' or 'in_top_k_error'

    erq_args =  list of values for k, 
               [1] for top_k for k = 1, 
               [1,5] for top_k for k = 1, 5
               - ignored for erq = 'mse'
    """
    self.erq = erq
    self.erq_args = erq_args
    self.erq_kwds = dict(erq_kwds)
    if self.erq is None: self.erq = DEFAULT_ERROR_QUOTIENT
    self.erq = Creation(self.erq)
    if self.erq == Creation('mse'):
      if len(erq_args):
        print("Warning: error quotient specifications for k are ignored for mean square errors")
        self.erq_args = ()
    else:
      if not len(self.erq_args):
        self.erq_args = [1],
        if 'name' not in erq_kwds:
          self.erq_kwds.update({'name': self.name + "/error_quotient"})
      elif type(self.erq_args[0]) is not list:
        self.erq_args = [self.erq_args[0]],

#-------------------------------------------------------------------------------
  def set_costfn(self, cfn = None, *cfn_args, **cfn_kwds):
    """
    cfn = 'mse' (mean squared error) or 'mce' (mean cross entropy)
    """
    self.cfn = cfn
    self.cfn_args = cfn_args
    self.cfn_kwds = dict(cfn_kwds)
    if self.cfn is None: self.cfn = DEFAULT_COST_FUNCTION
    if 'name' not in self.cfn_kwds:
      self.cfn_kwds.update({'name': self.name + "/cost"})

#-------------------------------------------------------------------------------
  def setup(self, ist = None, gst = None, skip_summaries = False, **kwds):

    if self.trainee is None: return

    # Setup learning rate, optimiser, and trainee network
    trainer.setup(self, ist, gst, True, **kwds)
    
    # Setup supervisor labels objects
    self._setup_labels()

    # Setup network output and error objects
    self._setup_errors()

    # Setup cost function
    self._setup_costfn()

    # Setup losses 
    self._setup_losses()

    # Setup gradient computations
    self._setup_gradients()

    # Setup parameter update operations
    self._setup_delta_ops()

    if skip_summaries: return

    # Setup summary scalars
    self._setup_scalars()

    # Setup summary distributions
    self._setup_distros()

#-------------------------------------------------------------------------------
  def _setup_labels(self):
    if self.dev is None:
      self.labels = Creation(self.lbl)(*self.lbl_args, **self.lbl_kwds)
    else:
      with Device(self.dev):
        self.labels = Creation(self.lbl)(*self.lbl_args, **self.lbl_kwds)

#-------------------------------------------------------------------------------
  def _setup_errors(self):
    self.hat_values = self.trainee.outnets
    self.errors = None
    if len(self.hat_values) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.hat_values = self.hat_values[0].ret_out()
    while type(self.hat_values) is tuple or type(self.hat_values) is list:
      if len(self.hat_values) != 1:
        raise ValueError('Supervisor class currently supports only a single unit stream output')
      else:
        self.hat_values = list(self.hat_values)[0]
    if not(len(self.erq_args[0])): # i.e. mse
      if self.dev is None:
        self.errors = [Creation(self.erq)(self.hat_values, self.labels, **self.erq_kwds)]
      else:
        with Device(self.dev):
          self.errors = [Creation(self.erq)(self.hat_values, self.labels, **self.erq_kwds)]
    else:
      self.errors = [None] * len(self.erq_args)
      # at the time of coding, "in_top_k" was not supported by GPU devices
      for i, k in enumerate(self.erq_args[0]):
        self.errors[i] = Creation(self.erq)(self.hat_values, self.labels, k,\
                                            name = self.name + "/error_quotient_" + str(k))
    return self.errors

#-------------------------------------------------------------------------------
  def _setup_costfn(self):
    self.arch_out = self.trainee.outnets
    self.cost = None
    if len(self.arch_out) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.trans_fn_out = self.arch_out[0].trans_fn
    if self.trans_fn_out in Logits_List: # pre-transfer-function value required
      self.arch_out = self.arch_out[0].arch_out
      if self.dev is None:
        self.cost = Creation(self.cfn)(self.arch_out, self.labels, *self.cfn_args,
                                       activation_fn = self.trans_fn_out, **self.cfn_kwds)
      else:
        with Device(self.dev):
          self.cost = Creation(self.cfn)(self.arch_out, self.labels, *self.cfn_args,
                                         activation_fn = self.trans_fn_out, **self.cfn_kwds)
    else: # we'll just use the hat values
      if self.dev is None:
        self.cost = Creation(self.cfn)(self.hat_values, self.labels, *self.cfn_args, **self.cfn_kwds)
      else:
        with Device(self.dev):
          self.cost = Creation(self.cfn)(self.hat_values, self.labels, *self.cfn_args, **self.cfn_kwds)
    return self.cost

#-------------------------------------------------------------------------------
  def _setup_losses(self):
    self.reg_losses = Keys('reg', scope = self.trainee.name)
    if self.dev is None:
      self.loss = Creation('add_ewise')([self.cost] + self.reg_losses, name = self.name + "/loss")
    else:
      with Device(self.dev):
        self.loss = Creation('add_ewise')([self.cost] + self.reg_losses, name = self.name + "/loss")
    return self.loss

#-------------------------------------------------------------------------------
  def _setup_gradients(self):
    # We calculate all parameter gradients, whether regimen-specified or not
    if self.dev is None:
      with Scope('var', self.name + "/", reuse = Flag('auto_reuse')):
        self.grad_and_vars = self.optimiser.compute_gradients(self.loss, var_list = self.variables)
    else:
      with Device(self.dev):
        with Scope('var', self.name + "/", reuse = Flag('auto_reuse')):
          self.grad_and_vars = self.optimiser.compute_gradients(self.loss, var_list = self.variables)
    self.gradients = [grad_and_vars[0] for grad_and_vars in self.grad_and_vars]
    self.gradient_names = [variable_name + "_gradients" for variable_name in self.variable_names]
    return self.gradients

#-------------------------------------------------------------------------------
  def _setup_delta_ops(self):
    # Parameter updates are regimen-dependent
    if not(self.n_regimens): # if not even a single regimen has been created
      self.add_regimen(DEFAULT_LEARNING_RATE)
    self.regimen_grad_and_vars = [None] * self.n_regimens
    self.delta_ops = [None] * self.n_regimens
    for i, _regimen in enumerate(self.regimens):
      self.regimen_grad_and_vars[i] = [self.grad_and_vars[ind] for ind in self.regimen_param_indices[i]]
      if self.dev is None:
        with variable_scope(self.opt_name + "/gradients/apply_regimen_"+str(i), reuse=Flag('auto_reuse')):
          self.delta_ops[i] = self.optimiser.apply_gradients(self.regimen_grad_and_vars[i], global_step = self.gst)
      else:
        with Device(self.dev):
          with variable_scope(self.opt_name + "/gradients/apply_regimen_"+str(i), reuse=Flag('auto_reuse')):
            self.delta_ops[i] = self.optimiser.apply_gradients(self.regimen_grad_and_vars[i], global_step = self.gst)
    return self.delta_ops

#-------------------------------------------------------------------------------
  def _setup_scalars(self, scalars = None, scalar_names = None,
                           train_scalars = None, train_scalar_names = None, 
                           test_scalars = None, test_scalar_names = None):
    trainer._setup_scalars(self, scalars, scalar_names)
    if train_scalars is None:
      train_scalars = [self.cost, self.loss]
      if self.errors is not None:
        train_scalars += self.errors
    if train_scalar_names is None:
      train_scalar_names = [self.name+"/COST_TRAIN", self.name+"/LOSS_TRAIN"]
      if self.errors is not None:
        if self.erq != Creation("in_top_k_error"):
          train_scalar_names.append(self.name+"/ERRQ_TRAIN")
        else:
          for i, err in enumerate(self.errors):
            train_scalar_names.append(self.name+"/ERRQ_TRAIN_"+str(self.erq_args[0][i]))
    if test_scalars is None:
      test_scalars = [self.cost, self.loss]
      if self.errors is not None:
        test_scalars += self.errors
    if test_scalar_names is None:
      test_scalar_names = [self.name+"/COST_TEST", self.name+"/LOSS_TEST"]
      if self.errors is not None:
        if self.erq != Creation("in_top_k_error"):
          test_scalar_names.append(self.name+"/ERRQ_TEST")
        else:
          for i, err in enumerate(self.errors):
            test_scalar_names.append(self.name+"/ERRQ_TEST_"+str(self.erq_args[0][i]))
    self.train_scalars, self.train_scalar_names = train_scalars, train_scalar_names
    self.test_scalars, self.test_scalar_names = test_scalars, test_scalar_names
    self.train_scalar_logs = []
    for scalar, scalar_name in zip(self.train_scalars, self.train_scalar_names):
      if scalar is not None:
        self.train_scalar_logs.append(Summary('scalar')(scalar_name, scalar))
    self.test_scalar_logs = []
    for scalar, scalar_name in zip(self.test_scalars, self.test_scalar_names):
      if scalar is not None:
        self.test_scalar_logs.append(Summary('scalar')(scalar_name, scalar))
    self.scalars_summary = [None] * (len(self.scalars) + len(self.test_scalars))
    self.summary_names = [None] * len(self.scalars_summary)
    for i, summary in enumerate(self.summary_names):
      if i < len(self.scalar_names):
        summary_name = self.scalar_names[i]
      else:
        summary_name = self.test_scalar_names[i - len(self.scalar_names)]
      self.summary_names[i] = summary_name.replace(self.name + "/", "")
    return self.scalars

#-------------------------------------------------------------------------------
  def set_feed_dict(self, is_training = False, feed_inputs = None, feed_labels = None):
    
    # This feed_dictionary supports inputs and labels
    feed_dict = {self.ist: is_training, self.inputs: feed_inputs, self.labels: feed_labels}

    # If training, updates learning_op
    if is_training: 
      self.feed_dict = feed_dict
      if self.session is not None:
        if self.using_regimen is None: self.using_regimen = -1
        if self.using_regimen < 0: self.use_regimen(0)
        self.progress[0] += 1                # one batch-update
        self.progress[1] += len(feed_inputs) # sum(batch_sizes)

    return feed_dict

#-------------------------------------------------------------------------------
  def train(self, *args):
    """
    Call using: self.train(inputs_data, labels_data)
    """
    if self.session is None:
      raise AttributeError("Cannot train without first invoking new_session")
    feed_dict = self.set_feed_dict(True, args[0], args[1])
    self.session.run([self.batch_size_op, self.delta_ops[self.using_regimen]], feed_dict = feed_dict)
    self.scalars_train = self.summarise()
    self.scalars_train = self.scalars_train[:len(self.scalars)]
    save_session = self.write_intervals[2]
    save_session = save_session if type(save_session) is bool else not(self.progress[0] % save_session)
    if save_session and self.write_dir is not None:
      self.save(self.write_dir + "/" + self.name)
    return self.scalars_train

#-------------------------------------------------------------------------------
  def summarise(self): # only relevent for training sets
    """
    Outputs numeric scalars
    """

    # Scalars 
    calc_scalars = self.write_intervals[0]
    calc_scalars = calc_scalars if type(calc_scalars) is bool else not(self.progress[0] % calc_scalars)
    if calc_scalars:
      scalars_num_log = self.session.run(self.scalars + self.train_scalars + self.scalar_logs + self.train_scalar_logs, 
                                         feed_dict = self.feed_dict)
      n_scalars = len(self.scalars) + len(self.train_scalars)
      scalars_num, scalars_log = scalars_num_log[:n_scalars], scalars_num_log[n_scalars:]
      self.scalars_summary[:len(self.scalars)] = scalars_num[:len(self.scalars)]
      self.add_logs(scalars_log)

    # Distros
    calc_distros = self.write_intervals[1]
    calc_distros = calc_distros if type(calc_distros) is bool else not(self.progress[0] % calc_distros)
    if calc_distros:
      distros_log = self.session.run(self.distro_logs, feed_dict = self.feed_dict)
      self.add_logs(distros_log)
    return self.scalars_summary

#-------------------------------------------------------------------------------
  def test(self, *args):
    """
    Call using: self.test(inputs_data, labels_data)
    """
    if self.session is None:
      raise AttributeError("Cannot test without first invoking new_session")
    feed_dict = self.set_feed_dict(False, args[0], args[1])
    scalars_num_log = self.session.run(self.test_scalars + self.test_scalar_logs, feed_dict = feed_dict)
    n_scalars = len(self.test_scalars)
    scalars_num, scalars_log = scalars_num_log[:n_scalars], scalars_num_log[n_scalars:]
    self.add_logs(scalars_log)
    self.scalars_summary[len(self.scalars):] = scalars_num
    summary_strs = [name + "=" + str(num) for name, num in zip(self.summary_names, self.scalars_summary)]
    return ', '.join(summary_strs)

#-------------------------------------------------------------------------------

