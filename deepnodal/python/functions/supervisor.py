"""
Supervisor module for Tensorflow. 
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.overseer import *

#-------------------------------------------------------------------------------
DEFAULT_OPTIMISER = 'sgd'
DEFAULT_COST_FUNCTION = 'mce'
DEFAULT_LABEL = 'tensor'
DEFAULT_LABEL_DTYPE = 'int64'
DEFAULT_ERROR_QUOTIENT = 'in_top_k_error'

#------------------------------------------------------------------------------- 
class supervisor (overseer):
  """
  A supervisor is an overseer with the benefit of labels. As a result, it should
  receive an expected output data set for each input data set. It uses a gradient-
  based optimiser (e.g. stochastic gradient descent) to update the parameters 
  within the work architecture according to pre-specified learning schedules based 
  on a cost or loss function. Optionally this can include additional 
  regularisation losses based on the magnitude of weight parameters.
  
  The convention adopted here is as follows:

  Loss = Cost + regularisation losses

  """
  def_name = 'supervisor'
  cfn = None                   # cost function
  lbl = None                   # labels
  erq = None                   # error quotient 
  gradients = None             # gradients
  grad_and_vars = None         # gradients and variables
  schedule_grad_and_vars = None  # grad_and_vars relevant to each schedule
  hatval = None                # object to compare labels for error calculations
  arch_out = None              # object to compare labels for cost calculations 
  cost = None                  # cost object
  cost_metric = None           # cost metric
  loss = None                  # loss object
  loss_metric = None           # loss metric
  errors = None                # list of errors
  error_metrics = None         # list of error metrics
  train_summary_str = None     # list of train summary string
  test_summary_str = None      # list of train summary string

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    overseer.__init__(self, name, dev)

#-------------------------------------------------------------------------------
  def set_work(self, work = None): # overloading trainer.set_work() to set defaults
    trainer.set_work(self, work)
    if self.lbl is None: self.set_labels()
    if self.cfn is None: self.set_costfn()
    if self.erq is None: self.set_errorq()

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
      self.lbl_kwds.update({'name': self.name + "/batch/labels"})
    
#-------------------------------------------------------------------------------
  def set_costfn(self, cfn = None, *cfn_args, **cfn_kwds):
    """
    cfn = 'mse' (mean squared error) or 'mce' (mean cross entropy)
    """
    self.cfn = cfn
    self.cfn_args = cfn_args
    self.cfn_kwds = dict(cfn_kwds)
    if self.cfn is None: self.cfn = DEFAULT_COST_FUNCTION
    if Creation(self.cfn) == Creation("nce"):
      pass

#-------------------------------------------------------------------------------
  def set_errorq(self, erq = None, *erq_args, **erq_kwds):
    """

    erq = 'mse' or 'in_top_k_error' or None

    erq_args =  list of values for k, 
               [1] for top_k for k = 1, 
               [1,5] for top_k for k = 1, 5
               - ignored for erq = 'mse'
    """
    self.erq = erq
    self.erq_args = erq_args
    self.erq_kwds = dict(erq_kwds)
    if self.erq is None: self.erq = DEFAULT_ERROR_QUOTIENT
    if Creation(self.erq) == Creation('mse'):
      if len(erq_args):
        print("Warning: error quotient specifications for k are ignored for mean square errors")
        self.erq_args = ()
    else:
      if not len(self.erq_args):
        self.erq_args = [1],
        if 'name' not in erq_kwds:
          self.erq_kwds.update({'name': self.name + "/metrics/error_quotient"})
      elif type(self.erq_args[0]) is not list:
        self.erq_args = [self.erq_args[0]],

#-------------------------------------------------------------------------------
  def __call__(self, ist = None, gst = None, skip_metrics = False, _called = True, **kwds):

    if self.work is None: return
    if self._opt is None: self.set_optimiser(DEFAULT_OPTIMISER)

    # Setup learning rate, optimiser, and work network
    overseer.__call__(self, ist, gst, True, False, **kwds)
    
    # Setup supervisor labels objects and metrics
    self._call_labels()
    self._call_metrics(skip_metrics)

    self.set_called(_called)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _call_metrics(self, skip_metrics = False):
    if skip_metrics: return

    # Setup hat values and call errors
    self._setup_hatval()
    self._call_errors()

    # Call cost and loss objects
    self._call_costfn()
    self._call_losses()

    # Call gradient computations and parameter update operations
    self._call_gradients()
    self._call_train_ops()

    # Call summary scalars and distributions
    self._call_summaries()

#-------------------------------------------------------------------------------
  def _setup_hatval(self):
    self.hatval = self.work.outnets
    if len(self.hatval) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.hatval = self.hatval[0].ret_out()
    while type(self.hatval) is tuple or type(self.hatval) is list:
      if len(self.hatval) != 1:
        raise ValueError('Supervisor class currently supports only a single unit stream output')
      else:
        self.hatval = list(self.hatval)[0]

#-------------------------------------------------------------------------------
  def _call_labels(self):
    if callable(Creation(self.lbl)):
      if self.dev is None:
        self.labels = Creation(self.lbl)(*self.lbl_args, **self.lbl_kwds)
      else:
        with Device(self.dev):
          self.labels = Creation(self.lbl)(*self.lbl_args, **self.lbl_kwds)
    else:
      self.labels = self.lbl

#-------------------------------------------------------------------------------
  def _call_errors(self):
    self.errors = None
    if self.erq is None: return self.errors
    if not len(self.erq_args): # i.e. mse
      if self.dev is None:
        with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
          self.errors = [Creation(self.erq)(self.hatval, self.labels, **self.erq_kwds)]
      else:
        with Device(self.dev):
          with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
            self.errors = [Creation(self.erq)(self.hatval, self.labels, **self.erq_kwds)]
      K = ['']
    else:
      self.errors = [None] * len(self.erq_args)
      K = [None] * len(self.erq_args[0])
      # at the time of coding, "in_top_k" was not supported by GPU devices
      for i, k in enumerate(self.erq_args[0]):
        self.errors[i] = Creation(self.erq)(self.hatval, self.labels, k,\
                                            name = self.name + "/metrics/error_quotient_" + str(k))
        K[i] = '_{}'.format(k)
      
    self.error_metrics = [None] * len(self.errors)
    for i in range(len(self.errors)):
      self.error_metrics[i] = self.add_metric()
      self.error_metrics[i].set_label('ERROR' + K[i], 'train', 'eval')
      self.error_metrics[i].__call__(self.errors[i])

    return self.errors

#-------------------------------------------------------------------------------
  def _call_costfn(self):
    self.arch_out = self.work.outnets
    self.cost = None
    if len(self.arch_out) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.trans_fn_out = Creation(self.arch_out[0].trans_fn)
    kwds = dict(self.cfn_kwds)
    if Creation(self.cfn) == Creation('mce') and self.trans_fn_out in Logits_List: # pre-transfer-function value required
      self.arch_out = self.arch_out[0].arch_out
      kwds.update({'name': self.name + "/metrics/cost"})
      if self.dev is None:
        self.cost = Creation(self.cfn)(self.arch_out, self.labels, *self.cfn_args,
                                       activation_fn = self.trans_fn_out, **kwds)
      else:
        with Device(self.dev):
          self.cost = Creation(self.cfn)(self.arch_out, self.labels, *self.cfn_args,
                                         activation_fn = self.trans_fn_out, **kwds)
    else: # we'll just use the hat values
      ndim_hatval, ndim_labels = len(Shape(self.hatval)), len(Shape(self.labels))
      if ndim_hatval == ndim_labels:
        if self.dev is None:
          with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
            self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
              self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
      elif ndim_hatval == 2 and ndim_labels == 1:
        # Here we attempt to create an interface that allows sparse labels to be converted to one-hot tensors
        if self.dev is None:
          self._labels = Creation('onehot')(self.labels, int(Shape(self.hatval)[-1]),
                                                   name=self.name+"/batch/labels/onehot")
        else:
          with Device(self.dev):
            self._labels = Creation('onehot')(self.labels, int(Shape(self.hatval)[-1]),
                                                     name=self.name+"/batch/labels/onehot")
        if self.dev is None:
          with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
            self.cost = Creation(self.cfn)(self.hatval, self._labels,
                                           *self.cfn_args, **self.cfn_kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
              self.cost = Creation(self.cfn)(self.hatval, self._labels,
                                             *self.cfn_args, **self.cfn_kwds)
      else:
        raise ValueError("Hat values and labels dimensionality incommensurate: " +
                         str(ndim_hatval) + "-D vs " + str(ndim_labels) + "-D")

    self.cost_metric = self.add_metric()
    self.cost_metric.set_label('COST', 'train', 'eval')
    self.cost_metric.__call__(self.cost)
    return self.cost

#-------------------------------------------------------------------------------
  def _call_losses(self):
    if self.reg_loss is None:
      if self.dev is None:
        self.loss = Creation('identity')(self.cost, name = self.name + "/metrics/loss")
      else:
        with Device(self.dev):
          self.loss = Creation('identity')(self.cost, name = self.name + "/metrics/loss")
    else:
      if self.dev is None:
        self.loss = Creation('add_ewise')([self.cost, self.reg_loss], name = self.name + "/metrics/loss")
      else:
        with Device(self.dev):
          self.loss = Creation('add_ewise')([self.cost,  self.reg_loss], name = self.name + "/metrics/loss")
    self.loss_metric = self.add_metric()
    self.loss_metric.set_label('LOSS', 'train', 'eval')
    self.loss_metric.__call__(self.loss)
    return self.loss

#-------------------------------------------------------------------------------
  def _call_gradients(self):
    # We calculate all parameter gradients, whether schedule-specified or not
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
  def _call_train_ops(self):
    # Parameter updates are schedule-dependent
    self.schedule_grad_and_vars = [None] * self.n_schedules
    self.lrate_ops = [None] * self.n_schedules # learning rate ops
    self.prepa_ops = [None] * self.n_schedules # preparatory ops
    self.delta_ops = [None] * self.n_schedules # delta parameter ops
    self.train_ops = [None] * self.n_schedules # training ops
    for i, _schedule in enumerate(self.schedules):
      self.schedule_grad_and_vars[i] = [self.grad_and_vars[ind] for ind in self.schedule_param_indices[i]]
      if self.dev is None:
        with variable_scope(self.name + "/schedules/apply_schedule_"+str(i), reuse=Flag('auto_reuse')):
          self.lrate_ops[i] = self.learning_rate.assign(self.schedules[i].learning_rate)
          self.prepa_ops[i] = Creation('combine')(self.lrate_ops[i], self.batch_size_op)
        with variable_scope(self.name + "/schedules/apply_schedule_"+str(i) + "/gradients", reuse=Flag('auto_reuse')):
          with Creation('deps')([self.prepa_ops[i]]):
            self.delta_ops[i] = self.optimiser.apply_gradients(self.schedule_grad_and_vars[i], 
                                                           global_step = self.gst)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/schedules/apply_schedule_"+str(i), reuse=Flag('auto_reuse')):
            self.lrate_ops[i] = self.learning_rate.assign(self.schedules[i].learning_rate)
            self.prepa_ops[i] = Creation('combine')(self.lrate_ops[i], self.batch_size_op)
          with variable_scope(self.name + "/schedules/apply_schedule_"+str(i) + "/gradients", reuse=Flag('auto_reuse')):
            with Creation('deps')([self.prepa_ops[i]]):
              self.delta_ops[i] = self.optimiser.apply_gradients(self.schedule_grad_and_vars[i], 
                                                             global_step = self.gst)
      self.train_ops[i] = self.delta_ops[i]
    return self.train_ops

#-------------------------------------------------------------------------------
  def _call_scalars(self):

    # Call trainer metrics
    self.train_scalars, self.train_scalar_labels, self.train_scalar_sublabels, \
        self.train_scalar_logs = overseer._call_scalars(self, 'train')

    # Call evaluation metrics
    self.eval_scalars, self.eval_scalar_labels, self.eval_scalar_sublabels, \
        self.eval_scalar_logs = overseer._call_scalars(self, 'eval')

    # Test metrics are the averages of evaluation metrics
    self.test_scalar_labels = [lbl.replace('EVAL', 'TEST') \
        for lbl in self.eval_scalar_labels]
    self.test_scalar_sublabels = [lbl.replace('EVAL', 'TEST') \
        for lbl in self.eval_scalar_sublabels]
    self.test_scalars = [None] * len(self.eval_scalars)
    self.test_scalar_logs = []
    for i in range(len(self.eval_scalars)):
      if self.eval_scalars[i] is not None:
        self.test_scalars[i] = Creation('var')(0., name=self.name+"/metrics/test_scalar_"+str(i))
        self.test_scalar_logs.append(Summary('scalar')(self.test_scalar_labels[i], self.test_scalars[i]))
    return self.scalars, self.scalar_labels, \
           self.scalar_sublabels, self.scalar_logs

#-------------------------------------------------------------------------------
  def set_feed_dict(self, is_training = False, feed_inputs = None, feed_labels = None):
    
    # This feed_dictionary supports inputs and labels
    feed_dict = {self.ist: is_training, self.inputs[0]: feed_inputs, self.labels: feed_labels}
    if self.session is None: return feed_dict

    # Default using_schedule if necessary
    if self.using_schedule is None: self.schedule = -1

    # If training, updates learning_op
    if is_training: 
      self.feed_dict = feed_dict
      self.use_schedule(max(0, self.using_schedule))
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
    self.session.run(self.train_ops[self.using_schedule], feed_dict = feed_dict)
    val_lbl = self.summarise()
    save_session = self.write_intervals[2]
    save_session = save_session if type(save_session) is bool else not(self.progress[0] % save_session)
    if save_session and self.write_dir is not None:
      self.save(self.write_dir + "/" + self.name)
    return val_lbl

#-------------------------------------------------------------------------------
  def summarise(self, force_log = False): # only relevent for training sets
    """
    Outputs numeric scalars
    """

    # Scalars 
    calc_scalars = self.write_intervals[0]
    calc_scalars = calc_scalars if type(calc_scalars) is bool else not(self.progress[0] % calc_scalars)
    scalars_val, sublabels = None, None
    if calc_scalars or force_log:
      scalars, labels, sublabels, logs = self.ret_scalar_group('train')
      scalars_val_log = self.session.run(scalars + logs, feed_dict = self.feed_dict)
      num_scalars = len(scalars)
      scalars_val, scalars_log = scalars_val_log[:num_scalars], scalars_val_log[num_scalars:]
      self._add_logs(scalars_log)
      summary_strs = [name + "=" + str(num) for name, num in zip(
        sublabels, scalars_val)]
      self.train_summary_str = ', '.join(summary_strs)

    # Distros
    calc_distros = self.write_intervals[1]
    calc_distros = calc_distros if type(calc_distros) is bool else not(self.progress[0] % calc_distros)
    if calc_distros or force_log:
      distros_log = self.session.run(self.distro_logs, feed_dict = self.feed_dict)
      self._add_logs(distros_log)
    return scalars_val, sublabels

#-------------------------------------------------------------------------------
  def test(self, *args, **kwds):
    """
    Call using: self.test(inputs_data, labels_data)
    """
    if self.session is None:
      raise AttributeError("Cannot test without first invoking new_session")
    split = 1 if 'split' not in kwds else kwds['split']
    scalars, labels, sublabels, logs = self.ret_scalar_group('eval')
    num_scalars = len(scalars)
    eval_scalars = np.empty([split, num_scalars], dtype = np.float32)
    arg0, arg1 = np.split(args[0], split), np.split(args[1], split)
    for i in range(split):
      feed_dict = self.set_feed_dict(False, arg0[i], arg1[i])
      eval_scalars[i, :] = self.session.run(scalars, feed_dict=feed_dict)
    test_scalars = np.mean(eval_scalars, axis=0)
    for i in range(num_scalars):
      self.session.run(self.test_scalars[i].assign(test_scalars[i]), feed_dict = {})
    scalars_log = self.session.run(self.test_scalar_logs)
    self._add_logs(scalars_log)
    summary_strs = [name + "=" + str(num) for name, num in zip(
      self.test_scalar_sublabels, test_scalars)]
    self.test_summary_str = ', '.join(summary_strs)
    return self.train_summary_str + "," + self.test_summary_str

#-------------------------------------------------------------------------------
