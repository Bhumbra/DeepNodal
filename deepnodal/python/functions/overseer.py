"""
Overseer module for TensorFlow. It is an abstract class with self.train as the
abstract method which must be defined by inheriting classes for instantiation.

A overseer is a trainer that can perform unsupervised or supervised learning 
with multiple learning regimen.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.functions.trainer import *

#-------------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 0.01

#-------------------------------------------------------------------------------
class overseer (trainer):
  """
  A overseer is an abstract class inheriting from trainer, associated with 
  a single optimiser to perform unsupervised or supervised training.
  In addition a overseer can use one or more training regimens for learning. 
  
  The class overseer does not calculate gradients.

  It is abstract, and inheriting classes must define self.train to be
  instantiable. The most immediate example of this is is the class supervisor.

  """
  def_name = 'overseer'
  regimes = None                   # list of training_regimes
  using_regime = None              # Python index of currently active regime
  regime_index = None              # Graph object of using_regime

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    trainer.__init__(self, name, dev)
    self.set_regimes()
    self.setup()

#-------------------------------------------------------------------------------
  def set_name(self, name = None):
    self.name = name if name is not None else self.def_name
    if self.regimes is None: return
    for i, _regimes in enumerate(self.regimes):
      _regimes.set_name(self.name+"/regime_"+str(i))

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    self.dev = dev
    if self.regimes is None: return
    for _regimes in self.regimes:
      _regimes.set_dev(self.dev)

#-------------------------------------------------------------------------------
  def set_global_step(self, gst = None):
    self.gst = gst
    if self.regimes is None: return
    for _regimes in self.regimes:
      _regimes.set_global_step(self.gst)

#-------------------------------------------------------------------------------
  def set_regimes(self, regimes = None):
    self.regimes = regimes
    if self.regimes is None:
      self.regimes = []
    self.n_regimes = len(self.regimes)
    self.use_regime() # initialises to -1

#-------------------------------------------------------------------------------
  def new_regime(self, lrn = None, *lrn_args, **lrn_kwargs):
    """
    New regime is created on the basis on learning rate specifications.
    The index of the new regime is returned.
    """
    self.regimes.append(regime(self.name + "/regimes/regime_" + str(len(self.regimes)), self.dev))
    self.regimes[-1].set_learning_rate(Creation(lrn), *lrn_args, **lrn_kwargs)
    self.n_regimes = len(self.regimes)
    return self.n_regimes - 1

#-------------------------------------------------------------------------------
  def set_regime(self, regime_index = None, dro = None, par = None):
    """
    dro is the dropout specification
    par is the parameter specification
    The affected regime class instance is returned.
    """
    self.regimes[regime_index].set_dropouts(dro)
    self.regimes[regime_index].set_parameters(par)
    return self.regimes[regime_index]

#-------------------------------------------------------------------------------
  def setup(self, ist = None, gst = None, skip_metrics = False):

    # Setup the regimes
    gst = self._setup_regimes(gst)

    # Setup all trainer objects except the metrics
    ist, gst = trainer.setup(self, ist, gst, True)

    # Collate the regimen parameter indices 
    self.regime_param_indices = [None] * self.n_regimes
    for i, _regime in enumerate(self.regimes):
      _regime.setup(self.gst)
      self.regime_param_indices[i] = self.work.ret_params(_regime.par, True)

    self._setup_metrics(skip_metrics)

    return self.ist, self.gst

#-------------------------------------------------------------------------------
  def _setup_regimes(self, gst = None): # this sets up the regime learning rate graph objects

    # Establish the global-step flag
    if self.gst is None: self.set_global_step(gst)
    if self.gst is None: self._setup_global_step()

    # Set up the regime index scalar - at the time of coding, not GPU-compatible
    """
    if self.dev is None:
      self.regime_index = Creation('var')(self.using_regime, trainable=False, name=self.name+"/regimes/index")
    else:
      with Device(self.dev):
        self.regime_index = Creation('var')(self.using_regime, trainable=False, name=self.name+"/regimes/index")
    """
    self.regime_index = Creation('var')(self.using_regime, trainable=False, name=self.name+"/regimes/index")

    # We need at least one regime
    if not(self.n_regimes):
      self.new_regime(DEFAULT_LEARNING_RATE)

    # Setup the regime instances
    for _regime in self.regimes: _regime.setup(self.gst)

    # Construct learning rate specifications suitable for regimens
    if self.n_regimes == -1:
      self.set_learning_rate('identity', self.regimes[0].learning_rate)
    else:
      self.set_learning_rate('var', 0)

    return self.gst

#-------------------------------------------------------------------------------
  def _setup_scalars(self, scalars = None, scalar_names = None):
    if scalars is None:
      scalars = [self.batch_size, self.learning_rate, self.regime_index]
    if scalar_names is None:
      scalar_names = [self.name + "/BATCH_SIZE",
                      self.name + "/LEARNING_RATE",
                      self.name + "/REGIME"]
    return trainer._setup_scalars(self, scalars, scalar_names)

#-------------------------------------------------------------------------------
  def set_feed_dict(self, is_training = False, feed_inputs = None):
    
    # This feed_dictionary supports only inputs
    feed_dict = {self.ist: is_training, self.inputs[0]: feed_inputs}
    if self.session is None: return feed_dict

    # Default using_regime if necessary
    if self.using_regime is None: self.regime = -1

    # If training, update batch_size and learning rate
    if is_training: 
      self.feed_dict = feed_dict
      self.use_regime(max(0, self.using_regime))
      self.progress[0] += 1                # one batch-update
      self.progress[1] += len(feed_inputs) # sum(batch_sizes)

    return feed_dict
   
#-------------------------------------------------------------------------------
  @abstractmethod
  def train(self, session = None, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
  def use_regime(self, using_regime = -1): # this updates regime index and learning rate
    # Returns whether the regime must be updated online
    if self.session is None:
      self.using_regime = using_regime
      return False
    
    update_regime = self.using_regime != using_regime

    # Update regime_index and run update operations if necessary
    if update_regime:
      # Regime learning rates and parameter list updates look after themselves
      # That just leaves the regime index and dropout
      self.using_regime = using_regime
      with Scope('var', self.name+"/regime_updates", reuse=True):
        op = self.regime_index.assign(self.using_regime)
        self.session.run(op)
      self.work.set_dropout(self.session, self.regimes[self.using_regime].dro)

    return update_regime
    
#-------------------------------------------------------------------------------

