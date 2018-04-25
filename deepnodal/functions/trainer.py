"""
Trainer module for Tensorflow. It is an abstract class with self.train as the
abstract method which must be defined by inheriting classes for instantiation.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.concepts.function import function
from deepnodal.functions.regimen import *
from deepnodal.structures.network import *

#-------------------------------------------------------------------------------
class trainer (function):
  """
  A trainer is a class containing a single optimiser to adopt different
  training regimens based on different learning rates, dropouts, and
  trainable parameters. It stops short of calculating gradients.

  It is abstract, and inheriting classes must define self.train to be
  instantiable. The most immediate example of this is the class supervisor.

  """
  def_name = 'trainer'
  def_write_intervals = [10, 1000, False] # write intervals [scalar, distro, model]
  gst = None                       # global_step
  ist = None                       # is training flag
  opt = None                       # optimiser specification
  trainee = None                   # network instance to train
  inputs = None                    # trainee input
  regimens = None                  # list of training_regimens
  regimen_index = None             # graph object of currently active regimen index
  learning_rate = None             # graph object currently active learning rate
  params = None                    # parameters of trainee as list of dictionaries
  outputs = None                   # outputs of trainee as a lsit
  using_regimen = None             # Python index of currently active regimen
  variables = None                 # entire list of parameter variables
  variable_names = None            # list of variable names          
  regimen_var_lists = None         # lists of variables for each regimen
  regimen_ind_lists = None         # compiled list of regimen parameter indices
  scalars = None                   # list of summary scalars
  distros = None                   # list of summary distributions
  progress = None                  # two-element list to allow save-model renewal
  write_dir = None                 # write directory
  logger = None                    # log writer creation
  saver = None                     # model saver creation
  session = None                   # training session
  gvi = None                       # global variables initialiser
  write_intervals = None           # see def_write_intervals

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    function.__init__(name, dev)
    self.set_global_step()
    self.set_is_training()
    self.set_trainee()
    self.set_optimiser()
    self.set_regimens()
    self.set_progress()
    self.set_write_intervals()
    self.setup()
    self.use_regimen() # this initialises using_regimen to -1

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev
    if self.regimens is None: return
    for _regimens in self.regimens:
      _regimens.set_dev(self.dev)

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst
    if self.regimens is None: return
    for _regimens in self.regimens:
      _regimens.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def set_is_training(self, ist = None):
    self.ist = ist

#-------------------------------------------------------------------------------
  def set_trainee(self, trainee = None):
    self.trainee = trainee
    if self.trainee is None: return
    if not isinstance(self.trainee, network):
      raise TypeError("Only suitable trainee is a network.")

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = None, *opt_args, **opt_kwds):
    self.opt = Creation(optim)
    self.opt_args = opt_args
    self.opt_kwds = dict(opt_kwds)
    if 'name' not in self.opt_kwds:
      self.opt_kwds = self.opt_kwds.update({'name':self.name + "/optimiser"})
    self.opt_name = self.opt_kwds['name']

#-------------------------------------------------------------------------------
  def set_regimens(self, regimens = None):
    self.regimens = regimens
    if self.regimens is None:
      self.regimens = []
    self.n_regimens = len(self.regimens)

#-------------------------------------------------------------------------------
  def set_progress(self, progress = None):
    self.progress = progress
    if self.progress is None:
      self.progress = [-1, 0] # [number of batch_updates, sum(batch_sizes)]
    return self.progress

#-------------------------------------------------------------------------------
  def new_regimen(self, lrate = None, *lrate_args, **lrate_kwargs):
    """
    New regimen is created on the basis on learning rate specifications.
    The index of the new regimen is returned.
    """
    self.regimens.append(regimen(self.name + "/regimen_" + str(i), self.dev))
    self.regimens[-1].set_learning_rate(lrate, *lrate_args, **lrate_kwargs)
    self.n_regimens = len(self.regimens)
    return self.n_regimens - 1

#-------------------------------------------------------------------------------
  def set_regimen_spec(self, regimen_index = None, dspec = None, pspec = None):
    """
    dspec is the dropout specification
    pspec is the parameter specification
    The affected regimen class instance is returned.
    """
    self.regimen[regimen_index].set_dropout_spec(dspec)
    self.regimen[regimen_index].set_parameter_spec(pspec)
    return self.regimen[regimen_index]

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
  def setup(self, gst = None, ist = None):
    if self.trainee is None: return 

    # Setup the trainer
    self._setup_learning(gst)
    self._setup_optimiser(ist)

    # Setup the trainee
    self._setup_variables()
    self._setup_outputs()

    # Setup the scalar and distribution summaries
    self._setup.scalars()
    self._setup_distros()

#-------------------------------------------------------------------------------
  def _setup_learning(self, gst = None): # this sets up batch_size, regimen index, and learning_rate
    if self.gst is None: self.set_global_step(gst)
    if self.gst is None: self._setup_global_step()

    # Here we set up the graph objects for basic learning parameters: batch_size, learning_rate, and regimen

    if self.trainee.inputs is None:
      raise AttributeError("Cannot setup trainer without network inputs specified.")
    if len(self.trainee.inp) != 1:
      raise ValueError("Current only single network input supported.")
    self.inputs = self.trainee.inp[0]

    # The learning_rate update-op must be done manually during the session
    if self.dev is None:
      self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch_size")
      self.batch_size_op = self.batch_size.assign(Shape(self.inputs)[0])
      self.learning_rate = Creation('var')(0., trainable=False, name=self.name+"/learning_rate")
      self.regimen_index = Creation('var')(self.using_regimen, trainable=False, name=self.name+"/regimen_index")
    else:
      with Device(self.dev):
        self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch_size")
        self.batch_size_op = self.batch_size.assign(Shape(self.inputs)[0])
        self.learning_rate = Creation('var')(0., trainable=False, name=self.name+"/learning_rate")
        self.regimen_index = Creation('var')(self.using_regimen, trainable=False, name=self.name+"/regimen_index")

#-------------------------------------------------------------------------------
  def _setup_global_step(self): # global_step is not device dependent
    self.gst = Creation('var')(0, trainable=False, name=self.name + "/global_step")
    self.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def _setup_optimiser(self, ist = None):
    if self.ist is None: self.set_is_training(ist)
    if self.ist is None: self._setup_is_training()
    self.trainee.set_is_training(self.ist)
    if self.dev is None:
      with Scope('var', self.name, reuse=Creation('auto_reuse')):
        self.optimiser = self.optim(*self.opt_args, learning_rate=self.learning_rate, **self.opt_kwds)
    else:
      with Device(self.dev):
        with Scope('var', self.name, reuse=Creation('auto_reuse')):
          self.optimiser = self.optim(*self.opt_args, learning_rate=self.learning_rate, **self.opt_kwds)

#-------------------------------------------------------------------------------
  def _setup_is_training(self):
    if self.dev is None:
      self.ist = Creation('var')(Dtype('bool'), trainable=False, name=self.name+"/is_training")
    else:
      with Device(self.dev):
        self.ist = Creation('var')(Dtype('bool'), trainable=False, name=self.name+"/is_training")

#-------------------------------------------------------------------------------
  def _setup_variables(self):
    """
    self.params is list of dictionaries in the form:
    self.param[i] = {self.variable_names[i]: self.variables}

    """
    # Access to variable objects requires first setting up the trainee (i.e. the entire network)
    self.trainee.setup(**kwds)

    # Set up variables
    self.params = self.trainee.ret_params()
    self.n_params = len(self.n_params)
    self.variable_names = [None] * self.n_params
    self.variables = [None] * self.n_params

    for i, param_dict in enumerate(self.params):
      self.variable_names[i] = list(param_dict)[0]
      self.variables = param_dict[self.variable_names[i]]

    self.regimen_ind_lists = [None] * self.n_regimens
    self.regimen_var_lists = [None] * self.n_regimens

    for i, _regimen in enumerate(self.regimens):
      self.regimen_var_lists[i], 
      self.regimen_ind_lists[i] = self.trainee.ret_params(_regimen.pspec, True)
    return self.variables

#-------------------------------------------------------------------------------
  def _setup_outputs(self):
    self.n_outputs = len(self.trainee.outputs)
    self.output_names = [None] * self.n_outputs
    self.outputs = [None] * self.n_outputs
    for i, output_dict in enumerate(self.trainee.outputs):
      self.output_names[i] = list(output_dict)[0]
      self.outputs[i] = output_dict[self.outputs_names[i]]
    return self.outputs

#-------------------------------------------------------------------------------
  def _setup_scalars(self, scalars = None, scalar_names = None):
    if scalars is None:
      scalars = [self.batch_size, self.regimen_index, self.learning_rate]
    if scalar_names is None:
      scalar_names = [self.name + "/BATCH_SIZE",
                      self.name + "/REGIMEN",
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
    self.distros, self.disto_names = distros, distro_names
    self.distro_logs = []
    for distro, distro_name in zip(self.distros, self.distro_names):
      if distro is not None:
        self.distro_logs.append(Summary('distro')(distro_name, distro))
    return self.distros

#-------------------------------------------------------------------------------
  def new_session(self, write_dir = None, *args, **kwds):
    if self.outputs is None: self.setup()

    # Setup the write_directory logger and saver.
    self._setup_write_dir(write_dir)
    self._setup_logger()
    self._setup_saver()
    self.session = Creation('session')(*args, **kwds)
    return self.session

#-------------------------------------------------------------------------------
  def init_variables(self, restorepoint = None):
    if self.session is None:
      raise AttributeError("Cannot initialise variables before calling new_session")
    with Scope('name'):
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
    self.saver.save(self.session, *args, global_step = self.gst, **kwargs)
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

    # Ideal point to confirm whether self.gvi has been setup and run
    if self.session is not None and self.gvi is None: self.init_variables()
    
    # This feed_dictionary supports only inputs
    feed_dict = {self.ist: is_training, self.inputs: feed_inputs}

    # If training, update batch_size and learning rate
    if is_training: 
      self.feed_dict = feed_dict
      if self.session is not None:
        if self.using_regimen is None: self.use_regimen(0)
        self.session.run(self.batch_size_op)
        op = self.learning_rate.assign(self.regimen[self.using_regimen].learning_rate)
        self.session.run(op)
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
    self.logger = Creation('logger')(self.write_dir, Creation('defaults')(), name = self.name + "/logger")

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
  def use_regimen(self, using_regimen = -1): # this updates regimen index and learning rate
    if self.session is None:
      self.using_regimen = using_regimen
      return self.using_regimen

    # Update regimen_index if necessary
    if self.using_regimen != using_regimen:
      self.using_regimen = using_regimen
      op = self.regimen_index.assign(self.using_regimen)
      self.session.run(op)

    # The batch-size and learning_rate is updated by the self.set_feed_dict.
    # The parameter list must be updated by inheriting classes.
    # That leaves dropout.
    self.trainee.set_dropout(self.session, self.regimen[self.using_regimen].dspec)
    return self.using_regimen
    
#-------------------------------------------------------------------------------

