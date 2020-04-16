"""
Trainer module for Tensorflow. It is an abstract class with self.train as the
abstract method which must be defined by inheriting classes for instantiation.

A trainer needs work (the associated network) according to unsupervised or
supervised learning rules with learning parameters.

The trainer module abstracts the tasks of session creation, variables handling,
logging, and saving.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
import csv
import numpy as np
from deepnodal.python.structures.network import *
from deepnodal.python.functions.recorder import *
from deepnodal.python.interfaces.calls import *
from deepnodal.python.cloud.gs import *


#-------------------------------------------------------------------------------
class trainer (recorder):
  """
  A trainer is a recorder with a single optimiser specification and provides an 
  abstract interface for handling a network's parameters, outputs, as well as 
  provision for training sessions, variable handling, loggers, and savers. 

  It is abstract, and inheriting classes must define self.train to be
  instantiable. The most immediate example of this is is the class supervisor, 
  via overseer.

  """
  def_name = 'trainer'
  def_write_intervals = [100, 10000, False] # write intervals [scalar, distro, model]
  progress = None                  # progress list: [number of batch_updates, sum(batch_sizes)] 
  gst = None                       # global_step
  ist = None                       # is training flag
  work = None                      # network instance to train
  inputs = None                    # work input
  params = None                    # parameters of work as list of dictionaries
  outputs = None                   # outputs of work as a list
  variables = None                 # entire list of parameter variables
  variable_names = None            # list of variable names          
  distros = None                   # list of summary distributions
  write_dir = None                 # write directory
  write_gcs = None                 # write google cloud storage
  logger = None                    # log writer creation
  saver = None                     # model saver creation
  session = None                   # training session
  gvi = None                       # global variables initialiser
  write_intervals = None           # see def_write_intervals
  metrics = None                   # list of metrics
  batch_size_metric = None         # batch size metric
  batch_size = None                # batch size
  batch_size_op = None             # batch size update op
  learning_rate_metric = None      # learning rate metric
  learning_rate = None             # learning rate

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    slave.__init__(self, name, dev)
    self.set_global_step()
    self.set_is_training()
    self.set_work()
    self.set_metrics()
    self.set_progress()
    self.set_write_intervals()

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst

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
  def __call__(self, ist = None, gst = None, skip_summaries = False, _called = True):
    if self.work is None: return 

    # Call is_training and network
    self._call_is_training(ist)
    self._call_inputs()

    # Call the trainer
    self._call_batch_size()
    self._call_learning_rate(gst)
    self._call_optimiser()

    # Call variables, regularisation, and architectural outputs
    self._call_variables()
    self._call_reguln()
    self._call_outputs()

    # Call scalars and distros
    self._call_summaries(skip_summaries)

    self.set_called(_called)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _call_summaries(self, skip_summaries = False):

    # Setup the scalar and distribution summaries
    if skip_summaries: return
    self._call_scalars()
    self._call_distros()

#-------------------------------------------------------------------------------
  def _call_is_training(self, ist = None):
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
  def _call_inputs(self): 

    # Setup the work and inputs
    self.work.__call__(self.ist)
    if self.work.inputs is None:
      raise AttributeError("Cannot setup trainer without network inputs specified.")
    if len(self.work._inp) != 1:
      raise ValueError("Current only single subnet network input supported.")

    self.inputs = self.work.ret_inp()
    return self.inputs

#-------------------------------------------------------------------------------
  def _call_batch_size(self):
    # At the time of coding, batch_size evaluation is not GPU-compatible
    self.batch_size_metric = self.add_metric('var', 0, trainable=False, 
        name=self.name+"/metrics/batch_size") 
    self.batch_size_metric.set_label("BATCH_SIZE", 'train')
    self.batch_size = self.batch_size_metric.__call__()
    with Scope('var', self.name+"/batch/batch_size_update", reuse=False):
      self.batch_size_op = self.batch_size_metric.call_assign(
          Creation('shape')(self.inputs[0])[0], reuse=None)

#-------------------------------------------------------------------------------
  def _call_learning_rate(self, gst = None): # this sets up self.learning_rate
    # Setup global_step first
    if self.gst is None: 
      if gst is None:
        self.__call__global_step()
      else:
        self.set_global_step(gst)

    # Call for the learning_rate - this is a copy-paste from regime.py but for 
    # good reason, since we want it to have the same capabilities even without
    # the multiple regimes provided by overseer.

    lrn = Creation(self._lrn)
    args = self._lrn_args
    kwds = dict(self._lrn_kwds)
    if 'name' not in kwds:
      kwds.update({'name': self.name + '/metrics/learning_rate'})

    if not callable(lrn):
      if self.dev is None:
        self.learning_rate = Creation('var')(lrn, *args, **kwds)
      else:
        with Device(self.dev):
          self.learning_rate = Creation('var')(lrn, *args, **kwds)
    else:
      if lrn == Creation('var'):
        if self.dev is None:
          self.learning_rate = lrn(*args, dtype=Dtype('float32'), **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*args, dtype=Dtype('float32'), **kwds)
      elif lrn == Creation('identity'):
        if self.dev is None:
          self.learning_rate = lrn(*args, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*args, **kwds)
      else:
        if self.dev is None:
          self.learning_rate = lrn(*args, global_step = self.gst, **kwds)
        else:
          with Device(self.dev):
            self.learning_rate = lrn(*lrn_args, global_step = self.gst, **kwds)
      self.learning_rate_metric = self.add_metric()
      self.learning_rate_metric.set_label('LEARNING_RATE', 'train')
      self.learning_rate_metric.__call__(self.learning_rate)

#-------------------------------------------------------------------------------
  def _call_global_step(self): # global_step is not device dependent
    self.gst = Creation('var')(0, trainable=False, 
                                  name=self.name+"/batch/global_step")
    self.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def _call_optimiser(self):
    kwds = dict(self._opt_kwds)
    if self.dev is None:
      with Scope('var', self.name, reuse=Flag('auto_reuse')):
        self.optimiser = Creation(self._opt)(*self._opt_args, learning_rate=self.learning_rate, **kwds)
    else:
      with Device(self.dev):
        with Scope('var', self.name, reuse=Flag('auto_reuse')):
          self.optimiser = Creation(self._opt)(*self._opt_args, learning_rate=self.learning_rate, **kwds)

#-------------------------------------------------------------------------------
  def _call_variables(self): # this creates no graph objects but inheriting classes might
    """
    self.params is list of dictionaries in the form:
    self.param[i] = {self.variable_names[i]: self.variables}
    """
    # Set up params
    self.params = self.work.ret_params()
    self.n_params = len(self.params)
    self.variable_names = [None] * self.n_params
    self.variables = [None] * self.n_params

    for i, param_dict in enumerate(self.params):
      self.variable_names[i] = list(param_dict)[0]
      self.variables[i] = param_dict[self.variable_names[i]]

    # Set up moments
    self.moments = self.work.ret_moments()
    self.n_moments = len(self.moments)
    self.moment_objects = [None] * self.n_moments
    self.moment_names = [None] * self.n_moments

    for i, moment_dict in enumerate(self.moments):
      self.moment_names[i] = list(moment_dict)[0]
      self.moment_objects[i] = moment_dict[self.moment_names[i]]

    # Set up updates
    self.updates = self.work.ret_updates()
    self.n_updates = len(self.updates)
    self.update_objects = [None] * self.n_updates
    self.update_names = [None] * self.n_updates

    for i, update_dict in enumerate(self.updates):
      self.update_names[i] = list(update_dict)[0]
      self.update_objects[i] = update_dict[self.update_names[i]]

    return self.variables

#-------------------------------------------------------------------------------
  def _call_reguln(self):
    self.reg_loss = None
    self.reg_losses = [param_and_loss[1] for param_and_loss in \
                       self.work.ret_reguln()['loss']]
    self.n_reg_losses= len(self.reg_losses)
    if not self.n_reg_losses: return self.reg_loss

    # Summate losses
    with Scope('var', self.name + "/batch/reg_loss", Flag("auto_reuse")):
      if self.dev is None:
        self.reg_loss = Creation('add_ewise')(self.reg_losses)
      else:
        with Device(self.dev):
          self.reg_loss = Creation('add_ewise')(self.reg_losses)

    return self.reg_loss

#-------------------------------------------------------------------------------
  def _call_outputs(self): # this creates no graph objects but inheriting classes might
    _outputs = self.work.ret_outputs()
    self.n_outputs = len(_outputs)
    self.output_names = [None] * self.n_outputs
    self.outputs = [None] * self.n_outputs
    for i, output_dict in enumerate(_outputs):
      self.output_names[i] = list(output_dict)[0]
      self.outputs[i] = output_dict[self.output_names[i]]
    return self.outputs

#-------------------------------------------------------------------------------
  def _call_distros(self, distros = None, distro_names = None):
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
  def set_write_dir(self, write_dir=None):
    """
    This creates no graph objects but should be called from new_session()
    """
    self.write_dir = write_dir
    self.write_gcs = None if write_dir[:5] != 'gs://' else GCS(self.write_dir)
    return self.write_dir

#-------------------------------------------------------------------------------
  def set_session(self, session = None):
    """
    This creates no graph objects but should be called from new_session()
    """
    self.session = session
    return self.session

#-------------------------------------------------------------------------------
  def call_session(self, write_dir=None, seed=None, *args, **kwds):

    # Initialise seed if specified
    if type(seed) is int:
      Seed(seed)
      seed = None

    # Call the write_directory logger and saver.
    if self.write_dir is None: self.set_write_dir(write_dir)
    if self.outputs is None: self.__call__()
    if self.logger is None: self._call_logger()
    if self.saver is None: self._call_saver()

    session = self._call_session(*args, **kwds)

    # Ideal point to confirm whether self.gvi has been setup and run
    if session is not None and self.gvi is None: 
      self._init_variables(seed)
    return session

#-------------------------------------------------------------------------------
  def _call_session(self, *args, **kwds):
    session = self.set_session(Creation('session')(*args, **kwds))
    return session

#-------------------------------------------------------------------------------
  def _call_logger(self, logger='logger'):
    self.logger = None
    if self.write_dir is None: return self.logger
    self.logger = Creation(logger)(self.write_dir, Creation('defaults')())

#-------------------------------------------------------------------------------
  def _call_saver(self, saver='saver'):
    # This will create a saver whether or not a write directory is given
    self.saver = None
    self.saver = Creation(saver)(name=self.name + "/saver")

#-------------------------------------------------------------------------------
  def _init_variables(self, seed=None):
    def _read_csv(path, *, delimiter='\t', newline='', permission='rt'):
      data = []
      with open(path, permission, newline=newline) as csv_read:
        csv_reader = csv.reader(csv_read, delimiter=delimiter)
        for row_read in csv_reader:
          data.append(row_read)
      return data
    if self.session is None:
      raise AttributeError("Cannot initialise variables before calling new_session")
    with Scope('name', self.name):
      self.gvi = Creation('gvi')()
    self.session.run(self.gvi)
    if isinstance(seed, str):
      if self.saver is None:
        raise AttributeError("Cannot load restore seed without saver created")
      self.saver.restore(self.session, seed)
      load_path = seed + '.tab'
      if self.write_gcs:
        data = self.write_gcs(_read_csv, load_path, 'read')
      else:
        data = _read_csv(load_path)
      self.progress = [int(data[0][0]), int(data[0][1])]
      np_random_state = data[1:]
      if len(np_random_state) == 5:
        np_random_state[0] = np_random_state[0][0]
        np_random_state[1] = np.array([int(element) for element in np_random_state[1]])
        np_random_state[2] = int(np_random_state[2][0])
        np_random_state[3] = int(np_random_state[3][0])
        np_random_state[4] = float(np_random_state[4][0])
        np.random.set_state(tuple(np_random_state))

#-------------------------------------------------------------------------------
  def save(self, *args, **kwds):
    def _write_csv(path, *, data=None, delimiter='\t', newline='', permission='wt'): 
      if data is None: return
      with open(path, permission, newline=newline) as csv_write:
        csv_writer = csv.writer(csv_write, delimiter=delimiter)
        for datum in data:
          if isinstance(datum, list):
            csv_writer.writerow(datum)
          else:
            csv_writer.writerow([str(datum)])
    """
    self.save(path) save self.session model to path
    """
    self.saver.save(self.session, *args, global_step=self.gst, **kwds)
    if not(len(args)): return
    # Could not successfully save batch progress to TensorFlow MetaGraph save points
    # so we save it manually.
    save_path = args[0] + "-" + str(int(self.gst.eval())) + ".tab"
    data = [[str(self.progress[0]), str(self.progress[1])]]
    np_random_state = list(np.random.get_state())
    np_random_state[1] = [str(element) for element in np_random_state[1]]
    data += np_random_state
    if self.write_gcs:
      self.write_gcs(_write_csv, save_path, 'write', data=data)
    else:
      _write_csv(save_path, data=data)

#-------------------------------------------------------------------------------
  def _add_logs(self, logs_str, flush=False):
    if self.logger is None: return
    if type(logs_str) is str: logs_str = [log_str]
    for log_str in logs_str:
      self.logger.add_summary(log_str, self.progress[1])
    if flush:
      self.logger.flush()

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
      scalars_obj_log = self.session.run(self.scalar_objects + self.scalars, feed_dict = self.feed_dict)
      n_scalars = len(self.scalars)
      scalars_obj, scalars_log = scalars_obj_log[:n_scalars], scalars_obj_log[n_scalars:]
      self._add_logs(scalars_logs)

    # Distros
    calc_distros = self.write_intervals[1]
    calc_distros = calc_distros if type(calc_distros) is bool else not(self.progress[0] % calc_distros)
    if calc_distros:
      distros_log = self.session.run(self.distro_logs, feed_dict = self.feed_dict)
      self._add_logs(distros_log)

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
  @abstractmethod
  def train(self, session = None, *args, **kwds):
    pass
    
#-------------------------------------------------------------------------------
  def ret_params(self, return_names = True):
    """
    Returns a list of parameters as numpy arrays
    if return_names is True, returns parameter_names, parameter_values
    """
    param_lbl = [None] * self._n_params
    param_val = [None] * self._n_params
    for i, param in enumerate(self._params):
      param_lbl[i] = list(param)[0]
      param_val[i] = param[param_lbl[i]].eval()

    if not(return_names): return param_val
    return param_lbl, param_val

#-------------------------------------------------------------------------------
