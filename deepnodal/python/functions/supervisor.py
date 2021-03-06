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
DEFAULT_CACHE_SIZE = 200

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
  pre = None                     # preapply function
  cfn = None                     # cost function
  lbl = None                     # labels
  erq = None                     # error quotient 
  gradients = None               # gradients
  grad_and_vars = None           # gradients and variables
  schedule_grad_and_vars = None  # grad_and_vars relevant to each schedule
  hatval = None                  # object to compare labels for error calculations
  pre_trans = None               # object to compare labels for cost calculations 
  cost = None                    # cost object
  cost_metric = None             # cost metric
  loss = None                    # loss object
  loss_metric = None             # loss metric
  errors = None                  # list of errors
  error_metrics = None           # list of error metrics
  lrate_ops = None               # learning-rate associated ops
  delta_ops = None               # regularisation-associated ops to gradients
  preta_ops = None               # operations before applying gradient updates
  preap_ops = None               # pre_apply ops
  apply_ops = None               # operations to apply gradients to variables
  posta_ops = None               # operations after applying gradient updates
  train_ops = None               # training operations
  train_summary_str = None       # summary string at train time
  test_summary_str = None        # summary string at test time
  eval_caches = None             # flag for continuous caching of training metric evaluations

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    overseer.__init__(self, name, dev)
    self.set_eval_caches()

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
    if not len(lbl_args) and 'dtype' not in self.lbl_kwds:
      self.lbl_kwds.update({'dtype': Dtype(DEFAULT_LABEL_DTYPE)})
    if 'name' not in self.lbl_kwds:
      self.lbl_kwds.update({'name': self.name + "/batch/labels"})
    
#-------------------------------------------------------------------------------
  def set_preap(self, pre=None, *pre_args, **pre_kwds):
    self.pre = pre
    self.pre_args = pre_args
    self.pre_kwds = pre_kwds
    if self.pre == 'clip_gnorm':
      assert self.pre_args, "Cannot clip_gnorm without specifying maximum"

#-------------------------------------------------------------------------------
  def set_costfn(self, cfn = None, *cfn_args, **cfn_kwds):
    """
    cfn = 'mse' (mean squared error) or 'mce' (mean cross entropy)
    """
    self.cfn = cfn
    self.cfn_args = cfn_args
    self.cfn_kwds = dict(cfn_kwds)
    if self.cfn is None: self.cfn = DEFAULT_COST_FUNCTION

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
  def set_eval_caches(self, eval_caches=True, cache_sizes=DEFAULT_CACHE_SIZE):
    self.eval_caches = eval_caches
    self.cache_sizes = cache_sizes

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

    # Call gradient computations and pre-update operations
    self._call_gradients()
    self._call_preta_ops()
     
    # Call update and post-update operations
    self._call_apply_ops()
    self._call_posta_ops()

    # Call summary scalars and distributions
    self._call_summaries()

#-------------------------------------------------------------------------------
  def _setup_hatval(self):
    self.hatval = self.work.outnets
    if len(self.hatval) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.hatval = self.hatval[0].ret_out()
    while type(self.hatval) is tuple or type(self.hatval) is list:
      if isinstance(self.hatval, tuple) and len(self.hatval) > 1:
        if self.dev is None:
          with Scope('var', self.name + "/metrics/hat_values/", reuse = Flag('auto_reuse')):
            self.hatval = Creation('con')([Creation('expand_dims')(hatval, axis=1) \
                                           for hatval in self.hatval], axis=1)
      elif len(self.hatval) != 1:
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
    if self.erq is False: return self.errors
    if not len(self.erq_args): # i.e. mse
      if self.dev is None:
        with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
          self.errors = [Creation(self.erq)(self.hatval, self.labels, **self.erq_kwds)]
      else:
        with Device(self.dev):
          with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
            self.errors = [Creation(self.erq)(self.hatval, self.labels, **self.erq_kwds)]
      K = ['']
    elif callable(self.erq):
      if self.dev is None:
        with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
          self.errors = [self.erq(self.hatval, self.labels, **self.erq_kwds)]
      else:
        with Device(self.dev):
          with Scope('var', self.name + "/metrics/error_quotient/", reuse = Flag('auto_reuse')):
            self.errors = [self.erq(self.hatval, self.labels, **self.erq_kwds)]
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
      self.error_metrics[i].set_label('ERROR' + K[i])
      self.error_metrics[i].set_groups({'train': self.cache_sizes,
                                        'batch': 0,
                                        'test': self.cache_sizes})
      self.error_metrics[i].__call__(self.errors[i])

    return self.errors

#-------------------------------------------------------------------------------
  def _call_costfn(self):
    self.cost = None
    if len(self.work.outnets) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.pre_trans = self.work.outnets[0].pre_trans
    self.trans_fn_out = Creation(self.work.outnets[0].trans_fn)
    kwds = dict(self.cfn_kwds)
    if Creation(self.cfn) == Creation("nce"):
      # TODO: We have to retrieve the weights and biases from the last layer
      pass
    elif Creation(self.cfn) in [Creation('mce'), Creation('sce')] \
      and self.trans_fn_out in Logits_List: # pre-transfer-function value required
      kwds.update({'name': self.name + "/metrics/cost"})
      if self.dev is None:
        self.cost = Creation(self.cfn)(self.pre_trans, self.labels, *self.cfn_args,
                                       activation_fn=self.trans_fn_out, **kwds)
      else:
        with Device(self.dev):
          self.cost = Creation(self.cfn)(self.pre_trans, self.labels, *self.cfn_args,
                                         activation_fn=self.trans_fn_out, **kwds)

    else: # we'll just use the hat values
      ndim_hatval, ndim_labels = len(Shape(self.hatval)), len(Shape(self.labels))
      if Creation(self.cfn) == Creation('nllc'):
        if ndim_hatval != ndim_labels + 1:
          raise ValueError("Hat values and labels dimensionality incommensurate: " +
                           str(ndim_hatval) + "-D vs " + str(ndim_labels) + "-D")
        if self.dev is None:
          with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
            self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
              self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
      elif ndim_hatval == ndim_labels:
        if self.dev is None:
          with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
            self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
              self.cost = Creation(self.cfn)(self.hatval, self.labels, *self.cfn_args, **self.cfn_kwds)
      elif (ndim_hatval == 2 and ndim_labels == 1) or (ndim_hatval == 3 and ndim_labels == 2):
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
      elif callable(self.cfn):
        if self.dev is None:
          with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
            self.cost = self.cfn(self.hatval, self.labels,
                                           *self.cfn_args, **self.cfn_kwds)
        else:
          with Device(self.dev):
            with Scope('var', self.name + "/metrics/cost", reuse=Flag('auto_reuse')):
              self.cost = self.cfn(self.hatval, self.labels,
                                             *self.cfn_args, **self.cfn_kwds)
      else:
        raise ValueError("Hat values and labels dimensionality incommensurate: " +
                         str(ndim_hatval) + "-D vs " + str(ndim_labels) + "-D")

    self.cost_metric = self.add_metric()
    self.cost_metric.set_label('COST')
    self.cost_metric.set_groups({'train': self.cache_sizes,
                                 'batch': 0,
                                 'test': self.cache_sizes})
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
    self.loss_metric.set_label('LOSS')
    self.loss_metric.set_groups({'train': self.cache_sizes,
                                 'batch': 0,
                                 'test': self.cache_sizes})
    self.loss_metric.__call__(self.loss)
    return self.loss

#-------------------------------------------------------------------------------
  def _call_gradients(self, skip_reg_grad=False):
    # Calculate all parameter gradients, whether schedule-specified or not
    if self.dev is None:
      with Scope('var', self.name + "/batch/", reuse = Flag('auto_reuse')):
        self.grad_and_vars = self.optimiser.compute_gradients(self.loss, var_list = self.variables)
    else:
      with Device(self.dev):
        with Scope('var', self.name + "/batch/", reuse = Flag('auto_reuse')):
          self.grad_and_vars = self.optimiser.compute_gradients(self.loss, var_list = self.variables)
    gradients = [_grad_and_vars[0] for _grad_and_vars in self.grad_and_vars]
    variables = [_grad_and_vars[1] for _grad_and_vars in self.grad_and_vars]
    grad_and_vars = [list(_grad_and_vars) for _grad_and_vars in self.grad_and_vars]
    self.gradient_names = [variable_name + "_gradients" for variable_name in self.variable_names]

    # Gradient delta functions associated with regularisation (e.g. weight-decay)
    self.reg_grad = self.work.ret_reguln()['grad']
    self.n_reg_grad = len(self.reg_grad)
    if skip_reg_grad or not self.n_reg_grad:
      self.gradients = gradients
      return self.gradients

    # Add gradients
    for (param, delta) in self.reg_grad:
      var_name, var = list(param.keys())[0], list(param.values())[0]
      index = variables.index(var)
      with Scope('var',  self.name + "/batch/reg_grad/"+var_name, reuse=Flag('auto_reuse')):
        if self.dev is None:
          grad_and_vars[index][0] = Creation('add')(gradients[index], delta)
        else:
          with Device(self.dev):
            grad_and_vars[index][0] = Creation('add')(gradients[index], delta)
    self.grad_and_vars = [tuple(_grad_and_vars) for _grad_and_vars in grad_and_vars]
    self.gradients = [grad_and_vars[0] for grad_and_vars in self.grad_and_vars]
    return self.gradients

#-------------------------------------------------------------------------------
  def _call_preta_ops(self):
    # Calls pre-training ops (i.e. update learning_rate and batch_size)
    self.lrate_ops = [None] * self.n_schedules # learning rate ops
    self.preta_ops = [None] * self.n_schedules # pre-training ops
    for i in range(self.n_schedules):
      if self.dev is None:
        with variable_scope(self.name + "/schedules/schedule_"+str(i), reuse=Flag('auto_reuse')):
          self.lrate_ops[i] = self.learning_rate.assign(self.schedules[i].learning_rate)
          self.preta_ops[i] = Creation('combine')(self.lrate_ops[i], self.batch_size_op)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/schedules/schedule_"+str(i), reuse=Flag('auto_reuse')):
            self.lrate_ops[i] = self.learning_rate.assign(self.schedules[i].learning_rate)
            self.preta_ops[i] = Creation('combine')(self.lrate_ops[i], self.batch_size_op)
    return self.preta_ops

#-------------------------------------------------------------------------------
  def _call_apply_ops(self):
    # Calls apply-gradient operations updates that are schedule-dependent
    self.schedule_grad_and_vars = [None] * self.n_schedules
    self.preap_ops = [None] * self.n_schedules # pre-apply ops
    self.apply_ops = [None] * self.n_schedules # apply gradient ops
    for i in range(self.n_schedules):
      self.schedule_grad_and_vars[i] = [self.grad_and_vars[ind] for ind in self.schedule_param_indices[i]]
      if self.pre:
        grad_and_vars = [list(_grad_and_vars) for _grad_and_vars in self.schedule_grad_and_vars[i]]
        grad = [_grad_and_vars[0] for _grad_and_vars in grad_and_vars]
        if self.dev is None:
          with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/"+self.pre, reuse=Flag('auto_reuse')):
            preap_grad, _ = Creation(self.pre)(grad, *self.pre_args, **self.pre_kwds)
        else:
          with Device(self.dev):
            with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/"+self.pre, reuse=Flag('auto_reuse')):
              preap_grad, _ = Creation(self.pre)(grad, *self.pre_args, **self.pre_kwds)
        for j in range(len(grad_and_vars)):
          grad_and_vars[j][0] = preap_grad[j]
          self.schedule_grad_and_vars[i][j] = tuple(grad_and_vars[j])
    for i in range(self.n_schedules):
      if self.dev is None:
        with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/apply", reuse=Flag('auto_reuse')):
          with Creation('deps')([self.preta_ops[i]]):
            self.apply_ops[i] = self.optimiser.apply_gradients(self.schedule_grad_and_vars[i], 
                                                               global_step = self.gst)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/apply/", reuse=Flag('auto_reuse')):
            with Creation('deps')([self.preta_ops[i]]):
              self.apply_ops[i] = self.optimiser.apply_gradients(self.schedule_grad_and_vars[i], 
                                                                 global_step = self.gst)

#-------------------------------------------------------------------------------
  def _call_posta_ops(self):
    # Calls post-training update ops (e.g. max_norm - TODO)
    self.posta_ops = []

    # Combine post-training ops with apply-ops
    self.train_ops = [None] * self.n_schedules
    for i in range(self.n_schedules):
      if self.dev is None:
        with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/train", reuse=Flag('auto_reuse')):
          self.train_ops[i] = Creation('combine')(self.apply_ops[i], self.update_objects, self.posta_ops)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/schedules/schedule_"+str(i) + "/train", reuse=Flag('auto_reuse')):
            self.train_ops[i] = Creation('combine')(self.apply_ops[i], self.update_objects, self.posta_ops)
    return self.train_ops

#-------------------------------------------------------------------------------
  def _call_scalars(self):

    # Call trainer metrics
    self.train_group = overseer._call_scalars(self, 'train')

    # Call batch metrics
    self.batch_group = overseer._call_scalars(self, 'batch')

    # Call test metrics
    self.test_group = overseer._call_scalars(self, 'test')

    return self.train_group, self.batch_group, self.test_group

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
  def train(self, *args, evaluate=None):
    """
    Call using: self.train(inputs_data, labels_data)
    """
    if self.session is None:
      raise AttributeError("Cannot train without first invoking new_session")
    feed_dict = self.set_feed_dict(True, *args)
    evaluations = None
    training_ops = self.train_ops[self.using_schedule]
    if evaluate is None and not self.eval_caches:
      self.session.run(self.train_ops[self.using_schedule], feed_dict=feed_dict)
    elif evaluate is None:
      metric_obj, metric_ops, _ = self.ret_obj_ops_means('train', reset=False)
      self.session.run([self.train_ops[self.using_schedule]] + 
                        metric_obj + metric_ops,
                        feed_dict=feed_dict)
    else:
      metric_obj, metric_ops, _ = self.ret_obj_ops_means('train', reset=False)
      evaluate = list(evaluate)
      evaluations = self.session.run([self.train_ops[self.using_schedule]] +
                                      metric_obj + metric_ops + evaluate,
                                      feed_dict=feed_dict)
      evaluations = evaluations[-len(evaluate):]
    summary = self.summarise()
    save_session = self.write_intervals[2]
    save_session = save_session if type(save_session) is bool \
                   else not(self.progress[0] % save_session)
    if save_session and self.write_dir is not None:
      self.save(self.write_dir + "/" + self.name)
    if evaluate is None:
      return summary
    return summary, evaluations

#-------------------------------------------------------------------------------
  def summarise(self, force_log = False): # only relevent for training batches
    """
    Outputs numeric scalars
    """
    # Scalars 
    calc_scalars = self.write_intervals[0]
    calc_scalars = calc_scalars if type(calc_scalars) is bool else not(self.progress[0] % calc_scalars)
    scalars_val, sublabels = None, None
    if calc_scalars or force_log:
      assert self.feed_dict[self.ist] is True, "Corrupted feed dictionary for batch sampling"
      train_str = ''
      if self.eval_caches:
        train_str = self.eval_log_scalars(self.session, 'train', feed_dict=self.feed_dict, flush=True)
      batch_str = self.eval_log_scalars(self.session, 'batch', feed_dict=self.feed_dict)
      self.train_summary_str = ', '.join([batch_str, train_str])

    # Distros
    calc_distros = self.write_intervals[1]
    calc_distros = calc_distros if type(calc_distros) is bool else not(self.progress[0] % calc_distros)
    if calc_distros or force_log:
      distros_log = self.session.run(self.distro_logs, feed_dict = self.feed_dict)
      self._add_logs(distros_log)
    if self.logger:
      if force_log or (calc_scalars or calc_distros):
        self.logger.flush()

    return self.train_summary_str

#-------------------------------------------------------------------------------
  def test(self, *args, skip_log=False, **kwds):
    """
    Call using: self.test(inputs_data, labels_data)
    """
    if self.session is None:
      raise AttributeError("Cannot test without first invoking new_session")
    split = 1 if 'split' not in kwds else kwds['split']
    metric_obj, metric_ops, _ = self.ret_obj_ops_means('test', reset=False)
    length = len(args[0])
    arg0, arg1 = np.split(args[0], split), np.split(args[1], split)
    self.test_summary_str = ''
    for i in range(split):
      feed_dict = self.set_feed_dict(False, arg0[i], arg1[i])
      self.session.run(metric_obj + metric_ops, feed_dict=feed_dict)
    self.test_summary_str = self.eval_log_scalars(self.session, 'test', feed_dict=feed_dict, flush=True)
    return ', '.join([self.train_summary_str, self.test_summary_str])

#-------------------------------------------------------------------------------
