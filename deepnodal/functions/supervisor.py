"""
Supervisor module for Tensorflow. 
"""

DEFAULT_OPTIMISER = 'sgd'
DEFAULT_COST_FUNCTION = 'mce'
DEFAULT_LABEL = 'var'
DEFAULT_LABEL_DTYPE = 'int64'

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.functions.trainer import *

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
  cfn = None          # cost function
  lbl = None          # labels
  err = None          # error quotient 
  output = None       # object to compare labels for error calculations
  arch_out = None     # object to compare labels for cost calculations

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    trainer.__init__(name, dev)
    self.set_optimiser()
    self.set_error_quotients()
    self.set_costfn()
    self.set_labels()

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = DEFAULT_OPTIMISER, *opt_args, **opt_kwds):
    """
    opt = 'sgd' or 'adam' etc...
    """
    self.opt = Creation(optim)
    self.opt_args = opt_args
    self.opt_kwds = dict(opt_kwds)
    if 'name' not in self.opt_kwds:
      self.opt_kwds = self.opt_kwds.update({'name':self.name + "/optimiser"})

#-------------------------------------------------------------------------------
  def set_error_quotients(self, erq = None, *erq_args, **erq_kwds):
    """

    erq = 'mse' or 'in_top_k_error'

    erq_args =  list of values for k, 
               [1] for top_k for k = 1, 
               [1,5] for top_k for k =1, 5
    """
    self.erq = erq
    self.erq_args = erq_args
    self.erq_kwds = dict(erq_kwds)
    if self.erq is None: return
    self.erq = Create(erq)
    if self.erq == Create('mse'):
      if len(erq_args):
        print("Warning: error quotient specifications for k are ignored for mean square errors")
        self.erq_args = ()
    else:
      if not len(self.erq_args):
        self.erq_args = [1],
        if 'name' not in erq_kwds:
          self.erq_kwds = self.erq_kwds.update({'name': self.name + "/error_quotient")
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
      self.cfn_kwds = self.cfn_kwds.update({'name': self.name + "/cost"})

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
    if not(len(self.lbl_args)):
      self.lbl_args = DEFAULT_LABEL_DTYPE,
    if 'shape' not in self.lbl_kwds:
      self.lbl_kwds = self.lbl.kwds.update({'shape': None})
    if 'name' not in self.lbl_kwds:
      self.lbl_kwds = self.lbl.kwds.update({'name': self.name + "/labels"})
    
#-------------------------------------------------------------------------------
  def setup(self, gst = None, ist = None, **kwds):

    # Setup learning rate, optimiser, and trainee network
    trainer.setup(self, gst, ist, **kwds)
    
    # Setup supervisor labels objects
    self._setup_labels()

    # Setup network output and error objects
    self._setup_errors()

    # Setup cost function
    self._setup_costfn()

    # Setup losses 
    self._setup_losses()

    # Setup parameter (gradients and variables) computations
    self._setup_params()

    # Setup summary scalars
    self._setup_scalrs()

    # Setup summary distributions
    self._setup_distrs()

#-------------------------------------------------------------------------------
  def _setup_labels(self):
    if self.dev is None:
      self.labels = Create(self.lbl)(*self.lbl_args, **self.lbl_kwds)
    else:
      with Device(self.dev):
        self.labels = Create(self.lbl)(*self.lbl_args, **self.lbl_kwds)

#-------------------------------------------------------------------------------
  def _setup_errors(self):
    self.output = self.trainee.outnets
    self.errors = None
    if len(self.output) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.output = self.output.ret_out()
    while type(self.output) is tuple or type(self.output) is list:
      if len(self.output) != 1:
        raise ValueError('Supervisor class currently supports only a single unit stream output')
      else:
        self.output = list(self.output)[0]
    if not(len(self.erq_args[0])): # i.e. mse
      if self.dev is None:
        self.errors = [Create(self.erq)(self.output, self.labels, **self.erq_kwds)]
      else:
        with Device(self.dev):
          self.errors = [Create(self.erq)(self.output, self.labels, **self.erq_kwds)]
    else:
      self.errors = [None] * len(self.erq_args)
      # at the time of coding, "in_top_k" was not supported by GPU devices
      for i, k in enumerate(self.erq_args[0]):
        self.errors[i] = Create(self.erq)(self.output, self.labels, k,
                                          name = self.name + "/error_quotient_k_" + str(k))
    return self.errors

#-------------------------------------------------------------------------------
  def _setup_costfn(self):
    self.arch_out = self.trainee.outnets
    self.cost = None
    if len(self.arch_out) != 1:
      raise ValueError('Supervisor class currently supports only a single unit stream output')
    self.arch_out = self.arch_out[0]
    self.trfn_out = self.arch_out.trans_fn
    if self.trfn_out in Logits_List: # pre-transfer-function value required
      self.arch_out = self.arch_out.arch_out
      if self.dev is None:
        self.cost = Create(self.cfn)(self.arch_out[0].arch_out, self.labels, self.trfn_out,
                                     **self.cfn_kwds)
      else:
        with Device(self.dev):
          self.cost = Create(self.cfn)(self.arch_out[0].arch_out, self.labels, self.trfn_out,
                                       **self.cfn_kwds)
    else:
      self.arch_out = self.arch_out.ret_out()
      while type(self.arch_out) is tuple or type(self.arch_output) is list:
        if len(self.arch_out) != 1:
          raise ValueError('Supervisor class currently supports only a single unit stream output')
        else:
          self.arch_out = list(self.arch_out)[0]
      if self.dev is None:
        self.cost = Create(self.cfn)(self.arch_out, self.labels, **cfn_kwds)
      else:
        with Device(self.dev):
          self.cost = Create(self.cfn)(self.arch_out, self.labels, **cfn_kwds)
    return self.cost

#-------------------------------------------------------------------------------
  def _setup_losses(self):
    self.reg_losses = Keys('reg', scope = self.trainee.name)
    if not(len(self.reg_losses)):
      if self.dev is None:
        self.loss = tf.identity(self.cost, name = self.name + "/loss")
      else:
        with Device(self.dev):
          self.loss = tf.identity(self.cost, name = self.name + "/loss")
    else:
      if self.dev is None:
        self.loss = Create('add_ewise')([self.cost] + self.reg_losses, name = self.name + "/loss")
      else:
        with Device(self.dev):
          self.loss = Create('add_ewise')([self.cost] + self.reg_losses, name = self.name + "/loss")

#-------------------------------------------------------------------------------
  def _setup_params(self, skipgrad = False):

    # Collate parameter lists
    self.params = [None] * self.n_regimens
    self.n_params = [None] * self.n_regimens
    for i, _regimen in enumerate(self.regimens):
      params_list = self.trainee.ret_params(_regimen.pspec)
      self.n_params[i] = len(params_list)
      self.params[i] = [None] * self.n_params
      for j in range(self.n_params[i]):
        param_name = list(params_list[0])
        self.params[i][j] = params_list[i][j]

    # Prepare gradient computations
    self.gradients = None
    self.deltas = None

    if skipgrad: # to allow external over-riding of gradient calculation
      return self.params

    self.gradients = [None] * self.regimens # this will also include the parameters
    self.deltas = [None] * self.regimens
    for i, _regimen in enumerate(self.regimens):
      if self.dev is None:
        with variable_scope(self.name + "/regimen_"+str(i), reuse=Flag('auto-reuse')):
          self.gradients[i] = self.optimiser.compute_gradients(self.loss, var_list=self.params[i])
          self.deltas[i] = self.optimiser.apply_gradients(self.gradients[i], global_step = self.gst)
      else:
        with Device(self.dev):
          with variable_scope(self.name + "/regimen_"+str(i), reuse=Flag('auto-reuse')):
            self.gradients[i] = self.optimiser.compute_gradients(self.loss, var_list=self.params[i])
            self.deltas[i] = self.optimiser.apply_gradients(self.gradients[i], global_step = self.gst)

    return self.gradients

#-------------------------------------------------------------------------------
  def _setup_scalrs(self, scalars = None, train_scalars = None, test_scalars = None):
    if scalars is None:
      scalars = [Summary('scalar')(self.name+"/BATCH_SIZE", self.batch_size),
                 Summary('scalar')(self.name+"/REGIMEN", self.regimen_index),
                 Summary('scalar')(self.name+"/LEARNING_RATE", self.learning_rate)]
    if train_scalars is None:
      train_scalars = [Summary('scalar')(self.name+"/COST_TRAIN", self.cost),
                       Summary('scalar')(self.name+"/LOSS_TRAIN", self.loss)]
      if self.errors is not None:
        if len(self.errors) == 1:
          train_scalars.append(Summary('scalar')(self.name+"ERRQ_TRAIN"))
        else:
          for i, err in enumerate(self.errors):
            train_scalars.append(Summary('scalar')(self.name+"ERRQ_TRAIN_"+str(self.erq_args[0][i])))

    if test_scalars is None:
      test_scalars = [Summary('scalar')(self.name+"/COST_TEST", self.cost),
                      Summary('scalar')(self.name+"/LOSS_TEST", self.loss)]
      if self.errors is not None:
        if len(self.errors) == 1:
          test_scalars.append(Summary('scalar')(self.name+"ERRQ_TEST"))
        else:
          for i, err in enumerate(self.errors):
            test_scalars.append(Summary('scalar')(self.name+"ERRQ_TEST_"+str(self.erq_args[0][i])))

#-------------------------------------------------------------------------------
  def train(self, *args, **kwds):
    pass

#-------------------------------------------------------------------------------

