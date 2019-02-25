"""
Overseer module for TensorFlow. It is an abstract class with self.train as the
abstract method which must be defined by inheriting classes for instantiation.

A overseer is a trainer that can perform unsupervised or supervised learning 
with multiple learning schedule.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.trainer import *
from deepnodal.python.functions.schedule import *

#-------------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 0.01

#-------------------------------------------------------------------------------
class overseer (trainer):
  """
  A overseer is an abstract class inheriting from trainer, associated with 
  a single optimiser to perform unsupervised or supervised training.
  In addition a overseer can use one or more training schedules for learning. 
  
  The class overseer does not calculate gradients.

  It is abstract, and inheriting classes must define self.train to be
  instantiable. The most immediate example of this is is the class supervisor.

  """
  def_name = 'overseer'
  schedules = None                   # list of training_schedules
  using_schedule = None              # Python index of currently active schedule
  schedule_index = None              # Graph object of using_schedule
  schedule_index_metric = None       # Metric for schedule_index

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    trainer.__init__(self, name, dev)
    self.set_schedules()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self.schedules is None: return
    for i, _schedules in enumerate(self.schedules):
      _schedules.set_name(self.name+"/schedule_"+str(i))

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev
    if self.schedules is None: return
    for _schedules in self.schedules:
      _schedules.set_dev(self.dev)

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst
    if self.schedules is None: return
    for _schedules in self.schedules:
      _schedules.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def set_schedules(self, schedules = None):
    self.schedules = schedules
    if self.schedules is None:
      self.schedules = []
    self.n_schedules = len(self.schedules)
    self.use_schedule() # initialises to -1

#-------------------------------------------------------------------------------
  def add_schedule(self, lrn = None, *lrn_args, **lrn_kwds):
    """
    New schedule is created on the basis on learning rate specifications.
    The index of the new schedule is returned.
    """
    self.schedules.append(schedule(self.name + "/schedules/schedule_" + str(len(self.schedules)), self.dev))
    self.schedules[-1].set_learning_rate(Creation(lrn), *lrn_args, **lrn_kwds)
    self.n_schedules = len(self.schedules)
    return self.n_schedules - 1

#-------------------------------------------------------------------------------
  def set_schedule(self, schedule_index = None, dro = None, par = None):
    """
    dro is the dropout specification
    par is the parameter specification
    The affected schedule class instance is returned.
    """
    self.schedules[schedule_index].set_dropouts(dro)
    self.schedules[schedule_index].set_parameters(par)
    return self.schedules[schedule_index]

#-------------------------------------------------------------------------------
  def __call__(self, ist = None, gst = None, skip_summaries = False, _called = True):

    # Call the schedules
    gst = self._call_schedules(gst)

    # Call all trainer objects except the metrics
    ist, gst = trainer.__call__(self, ist, gst, True, False)

    # Collate the schedule parameter indices 
    self.schedule_param_indices = [None] * self.n_schedules

    for i, _schedule in enumerate(self.schedules):
      _schedule.__call__(self.gst)
      self.schedule_param_indices[i] = self.work.ret_params(_schedule.par, True)

    if not skip_summaries:
      self._call_summaries(skip_summaries)

    self.set_called(_called)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _call_schedules(self, gst = None): # this sets up the schedule learning rate graph objects

    # Establish the global-step flag
    if self.gst is None: self.set_global_step(gst)
    if self.gst is None: self._call_global_step()

    # Set up the schedule index scalar - at the time of coding, not GPU-compatible
    self.schedule_index_metric = self.add_metric('var', self.using_schedule, trainable=False, 
                                 name=self.name+"/schedules/index")
    self.schedule_index_metric.set_label("SCHEDULE_INDEX", 'train')
    self.schedule_index = self.schedule_index_metric.__call__()

    # We need at least one schedule
    if not(self.n_schedules):
      self.add_schedule(DEFAULT_LEARNING_RATE)

    # Call the schedule instances
    for _schedule in self.schedules: _schedule.__call__(self.gst)

    # Construct learning rate specifications suitable for schedules
    if self.n_schedules == -1:
      self.set_learning_rate('identity', self.schedules[0].learning_rate)
    else:
      self.set_learning_rate('var', 0)

    return self.gst

#-------------------------------------------------------------------------------
  def set_feed_dict(self, is_training = False, feed_inputs = None):
    
    # This feed_dictionary supports only inputs
    feed_dict = {self.ist: is_training, self.inputs[0]: feed_inputs}
    if self.session is None: return feed_dict

    # Default using_schedule if necessary
    if self.using_schedule is None: self.schedule = -1

    # If training, update batch_size and learning rate
    if is_training: 
      self.feed_dict = feed_dict
      self.use_schedule(max(0, self.using_schedule))
      self.progress[0] += 1                # one batch-update
      self.progress[1] += len(feed_inputs) # sum(batch_sizes)

    return feed_dict
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def train(self, session = None, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
  def use_schedule(self, using_schedule = -1): # this updates schedule index and learning rate
    # Returns whether the schedule must be updated online
    if self.session is None:
      self.using_schedule = using_schedule
      return False
    
    update_schedule = self.using_schedule != using_schedule

    # Update schedule_index and run update operations if necessary
    if update_schedule:
      # Regime learning rates and parameter list updates look after themselves
      # That just leaves the schedule index and dropout
      self.using_schedule = using_schedule
      with Scope('var', self.name+"/schedule_updates", reuse=True):
        op = self.schedule_index.assign(self.using_schedule)
        self.session.run(op)
      self.work.set_dropout(self.session, self.schedules[self.using_schedule].dro)

    return update_schedule
    
#-------------------------------------------------------------------------------

