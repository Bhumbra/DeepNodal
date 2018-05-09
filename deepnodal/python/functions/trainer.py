"""
Slave module for Tensorflow. It is an abstract class with self.train as the
abstract method which must be defined by inheriting classes for instantiation.

A trainer needs work (the associated network) according to unsupervised or
supervised learning rules with learning parameters.

The trainer module abstracts the tasks of session creation, variables handling,
logging, and saving.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.regime import *
from deepnodal.python.structures.network import *
import csv

#-------------------------------------------------------------------------------
class trainer (slave):
  """
  A trainer is a slave with a single optimiser specification and provides an 
  abstract interface for handling a network's parameters, outputs, as well as 
  provision for training sessions, variable handling, loggers, and savers. 

  It is abstract, and inheriting classes must define self.train to be
  instantiable. The most immediate example of this is is the class supervisor, 
  via overseer.

  """
  def_name = 'trainer'
  def_write_intervals = [10, 1000, False] # write intervals [scalar, distro, model]
  gst = None                       # global_step
  ist = None                       # is training flag
  work = None                      # network instance to train
  inputs = None                    # work input
  params = None                    # parameters of work as list of dictionaries
  outputs = None                   # outputs of work as a lsit
  variables = None                 # entire list of parameter variables
  variable_names = None            # list of variable names          
  scalars = None                   # list of summary scalars
  distros = None                   # list of summary distributions
  write_dir = None                 # write directory
  logger = None                    # log writer creation
  saver = None                     # model saver creation
  session = None                   # training session
  gvi = None                       # global variables initialiser
  write_intervals = None           # see def_write_intervals

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    slave.__init__(self, name, dev)
    self.set_global_step()
    self.set_is_training()
    self.set_work()
    self.set_write_intervals()

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst
    if self.regimes is None: return
    for _regimes in self.regimes:
      _regimes.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def set_is_training(self, ist = None):
    self.ist = ist

#-------------------------------------------------------------------------------
  def set_work(self, work = None):
    self.work = work
    if self.work is None: return
    if not isinstance(self.work, network) and not issubclass(self.work, network):
      raise TypeError("Only suitable work is a network.")
    if self.dev is not None:
      self.work.set_dev(self.dev)

#-------------------------------------------------------------------------------
  def set_progress(self, progress = None):
    self.progress = progress
    if self.progress is None:
      self.progress = [-1, 0] # [number of batch_updates, sum(batch_sizes)]
    return self.progress

#-------------------------------------------------------------------------------
  def set_write_intervals(self, write_intervals = None):
    """
    write_intervals = [interval for scalar update,
                       interval for distro update,
                       interval for model update]
    """
    self.write_intervals = write_intervals
    if self.write_intervals is None: self.write_intervals = self.def_write_intervals

#-------------------------------------------------------------------------------
  def setup(self, ist = None, gst = None, skip_metrics = False):
    if self.work is None: return 

    # Setup variables and outputs
    self._setup_is_training(ist)
    self._setup_inputs()
    self._setup_variables()
    self._setup_outputs()

    # Setup the trainer
    self._setup_batch_size()
    self._setup_learning_rate(gst)
    self._setup_optimiser()
    self._setup_metrics(skip_metrics)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _setup_metrics(self, skip_metrics = False):
    # Setup the scalar and distribution summaries
    if skip_metrics: return
    self._setup_scalars()
    self._setup_distros()

#-------------------------------------------------------------------------------
  def _setup_is_training(self, ist = None):
    if self.ist is None:
      if ist is None:
        pass
      else:
        return self.set_is_training(ist)
    else:
      return self.ist
    if self.dev is None:
      self.ist = Creation('tensor')(Dtype('bool'), name=self.name+"/batch/is_training")
    else:
      with Device(self.dev):
        self.ist = Creation('tensor')(Dtype('bool'), name=self.name+"/batch/is_training")
    return self.ist

#-------------------------------------------------------------------------------
  def _setup_inputs(self):

    # Setup the work and inputs
    self.work.setup(self.ist)
    if self.work.inputs is None:
      raise AttributeError("Cannot setup trainer without network inputs specified.")
    if len(self.work.inp) != 1:
      raise ValueError("Current only single subnet network input supported.")
    self.inputs = self.work.inp
    return self.inputs

#-------------------------------------------------------------------------------
  def _setup_variables(self): # this creates no graph objects
    """
    self.params is list of dictionaries in the form:
    self.param[i] = {self.variable_names[i]: self.variables}
    """

    # Set up variables
    self.params = self.work.ret_params()
    self.n_params = len(self.params)
    self.variable_names = [None] * self.n_params
    self.variables = [None] * self.n_params

    for i, param_dict in enumerate(self.params):
      self.variable_names[i] = list(param_dict)[0]
      self.variables[i] = param_dict[self.variable_names[i]]

    return self.variables

#-------------------------------------------------------------------------------
  def _setup_outputs(self): # this creates no graph objects
    self.n_outputs = len(self.work.outputs)
    self.output_names = [None] * self.n_outputs
    self.outputs = [None] * self.n_outputs
    for i, output_dict in enumerate(self.work.outputs):
      self.output_names[i] = list(output_dict)[0]
      self.outputs[i] = output_dict[self.output_names[i]]
    return self.outputs

#-------------------------------------------------------------------------------
  def _setup_batch_size(self):
    # At the time of coding, batch_size evaluation is not GPU-compatible
    """
    if self.dev is None:
      self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch/batch_size")
      with Scope('var', self.name+"/batch/batch_size_update", reuse=False):
        self.batch_size_op = self.batch_size.assign(Creation('shape')(self.inputs[0])[0])
    else:
      with Device(self.dev):
        self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch/batch_size")
        with Scope('var', self.name+"/batch/batch_size_update", reuse=False):
          self.batch_size_op = self.batch_size.assign(Creation('shape')(self.inputs[0])[0])
    """
    self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch/batch_size")
    with Scope('var', self.name+"/batch/batch_size_update", reuse=False):
      self.batch_size_op = self.batch_size.assign(Creation('shape')(self.inputs[0])[0])

#-------------------------------------------------------------------------------
  def _setup_learning_rate(self, gst = None): # this sets up self.learning_rate
    # Setup global_step first
    if self.gst is None: 
      if gst is None:
        self._setup_global_step()
      else:
        self.set_global_step(gst)

    # Setup the learning_rate - this is a copy-paste from regime.py but for good reason

    lrn = Creation(self.lrn)
    kwds = dict(self.lrn_kwds)
    if 'name' not in kwds:
      kwds.update({'name': self.name + '/learning_rate'})

    if not callable(lrn):
      if self.dev is None:
        self.learning_rate = Creation('var')(lrn, *self.lrn_args, **kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(lrn, *self.lrn_args, **kwds)
    else:
      if lrn == Creation('var'):
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, dtype=Dtype('float32'), **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, dtype=Dtype('float32'), **kwds)
      elif lrn == Creation('identity'):
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, **kwds)
      else:
        if self.dev is None:
          self.learning_rate = lrn(*self.lrn_args, global_step = self.gst, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*self.lrn_args, global_step = self.gst, **kwds)

#-------------------------------------------------------------------------------
  def _setup_global_step(self): # global_step is not device dependent
    self.gst = Creation('var')(0, trainable=False, name=self.name + "/global_step")
    self.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def _setup_optimiser(self):
    kwds = dict(self.opt_kwds)
    if self.dev is None:
      with Scope('var', self.name, reuse=Flag('auto_reuse')):
        self.optimiser = Creation(self.opt)(*self.opt_args, learning_rate=self.learning_rate, **kwds)
    else:
      with Device(self.dev):
        with Scope('var', self.name, reuse=Flag('auto_reuse')):
          self.optimiser = Creation(self.opt)(*self.opt_args, learning_rate=self.learning_rate, **kwds)

#-------------------------------------------------------------------------------
  def _setup_scalars(self, scalars = None, scalar_names = None):
    if scalars is None:
      scalars = [self.batch_size, self.learning_rate]
    if scalar_names is None:
      scalar_names = [self.name + "/BATCH_SIZE",
                      self.name + "/LEARNING_RATE"]
    self.scalars, self.scalar_names = scalars, scalar_names
    self.scalar_logs = []
    for scalar, scalar_name in zip(self.scalars, self.scalar_names):
      if scalar is not None:
        self.scalar_logs.append(Summary('scalar')(scalar_name, scalar))
    return self.scalars

#-------------------------------------------------------------------------------
  def _setup_distros(self, distros = None, distro_names = None):
    if distros is None:
      distros = self.outputs + self.variables + self.gradients 
    if distro_names is None:
      distro_names = self.output_names + self.variable_names + self.gradient_names
    self.distros, self.distro_names = distros, distro_names
    self.distro_logs = []
    for distro, distro_name in zip(self.distros, self.distro_names):
      if distro is not None:
        self.distro_logs.append(Summary('distro')(distro_name, distro))
    return self.distros

#-------------------------------------------------------------------------------
  def set_session(self, session = None):
    self.session = session
    return self.session

#-------------------------------------------------------------------------------
  def new_session(self, write_dir = None, *args, **kwds):
    if self.outputs is None: self.setup()

    # Setup the write_directory logger and saver.
    self._setup_write_dir(write_dir)
    self._setup_logger()
    self._setup_saver()
    session = self.set_session(Creation('session')(*args, **kwds))

    # Ideal point to confirm whether self.gvi has been setup and run
    if session is not None and self.gvi is None: self.init_variables()
    return session

#-------------------------------------------------------------------------------
  def init_variables(self, restorepoint = None):
    if self.session is None:
      raise AttributeError("Cannot initialise variables before calling new_session")
    with Scope('name', self.name):
      self.gvi = Creation('gvi')()
    self.session.run(self.gvi)
    if restorepoint is not None:
      if self.saver is None:
        raise AttributeError("Cannot load restore point without saver created")
      self.saver.restore(self.session, restorepoint)
      load_path = restorepoint + '.tab'
      with open(load_path, 'rt') as tab_file:
        tab_reader = csv.read(tab_file, delimiter = '\t')
        progress = []
        for row in tab_reader:
          if not(len(progress)):
            progress = row
      self.progress = [int(progress[0]), int(progress[1])]

#-------------------------------------------------------------------------------
  def save(self, *args, **kwds):
    """
    self.save(path) save self.session model to path
    """
    self.saver.save(self.session, *args, global_step = self.gst, **kwds)
    if not(len(args)): return
    # Could not successfully save batch progress to TensorFlow MetaGraph save points
    # so we save it manually.
    save_path = args[0] + "-" + str(int(self.gst.eval())) + ".tab"
    progress = [str(self.progress[0]), str(self.progress[1])]
    with open(save_path, 'wt', newline="") as tab_file:
      tab_writer = csv.writer(tab_file, delimiter = '\t')
      tab_writer.writerow(progress)

#-------------------------------------------------------------------------------
  def add_logs(self, logs_str):
    if self.logger is None: return
    if type(logs_str) is str: logs_str = [log_str]
    for log_str in logs_str:
      self.logger.add_summary(log_str, self.progress[1])

#-------------------------------------------------------------------------------
  def summarise(self):
    """
    Outputs numeric scalars
    """
    scalars_num = [None] * len(self.scalars)

    # Scalars 
    calc_scalars = self.write_intervals[0]
    calc_scalars = calc_scalars if type(calc_scalars) is bool else not(self.progress[0] % calc_scalars)
    if calc_scalars:
      scalars_num_log = self.session.run(self.scalars + self.scalar_logs, feed_dict = self.feed_dict)
      n_scalars = len(self.scalars)
      scalars_num, scalars_log = scalars_num_log[:n_scalars], scalars_num_log[n_scalars:]
      self.add_logs(scalars_logs)

    # Distros
    calc_distros = self.write_intervals[1]
    calc_distros = calc_distros if type(calc_distros) is bool else not(self.progress[0] % calc_distros)
    if calc_distros:
      distros_log = self.session.run(self.distro_logs, feed_dict = self.feed_dict)
      self.add_logs(distros_log)

    return scalars_num

#-------------------------------------------------------------------------------
  def set_feed_dict(self, is_training = False, feed_inputs = None):
    
    # This feed_dictionary supports only inputs
    feed_dict = {self.ist: is_training, self.inputs[0]: feed_inputs}
    if self.session is None: return feed_dict

    # If training, update batch_size and learning rate
    if is_training: 
      self.feed_dict = feed_dict
      self.progress[0] += 1                # one batch-update
      self.progress[1] += len(feed_inputs) # sum(batch_sizes)

    return feed_dict
   
#-------------------------------------------------------------------------------
  def _setup_write_dir(self, write_dir = None):
    """
    This creates no graph objects but should be called from new_session()

    """
    self.write_dir = write_dir

#-------------------------------------------------------------------------------
  def _setup_logger(self):
    self.logger = None
    if self.write_dir is None: return self.logger
    self.logger = Creation('logger')(self.write_dir, Creation('defaults')())

#-------------------------------------------------------------------------------
  def _setup_saver(self, saver = 'saver'):
    # This will create a saver whether or not a write directory is given
    self.saver = None
    self.saver = Creation(saver)(name = self.name + "/saver")

#-------------------------------------------------------------------------------
  @abstractmethod
  def train(self, session = None, *args, **kwds):
    pass
    
#-------------------------------------------------------------------------------
  def ret_params(self, return_names = True):
    """
    Returns a list of parameters as numpy arrays
    if return_names is True, returns parameter_names, parameter_values
    """
    param_lbl = [None] * self.n_params
    param_val = [None] * self.n_params
    for i, param in enumerate(self.params):
      param_lbl[i] = list(param)[0]
      param_val[i] = param[param_lbl[i]].eval()

    if not(return_names): return param_val
    return param_lbl, param_val

#-------------------------------------------------------------------------------

