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
  gst = None                    # global_step
  ist = None                    # is training flag
  opt = None                    # optimiser specification
  trainee = None                # network instance to train
  regimens = None               # list of training_regimens
  train_regimen = None          # Python index of currently active regimen index
  regimen_index = None          # graph object of currently active regimen index
  learning_rate = None          # graph object currently active learning rate

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    function.__init__(name, dev)
    self.set_global_step()
    self.set_is_training()
    self.set_trainee()
    self.set_optimiser()
    self.set_regimens()
    self.setup()

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
      raise(TypeError("Only suitable trainee is a network.")

#-------------------------------------------------------------------------------
  def set_optimiser(self, opt = None, *opt_args, **opt_kwds):
    self.opt = Creation(optim)
    self.opt_args = opt_args
    self.opt_kwds = dict(opt_kwds)
    if 'name' not in self.opt_kwds:
      self.opt_kwds = self.opt_kwds.update({'name':self.name + "/optimiser"})

#-------------------------------------------------------------------------------
  def set_train_regimen(self, train_regimen = None, *args):
    self.train_regimen = train_regimen
    return self.train_regimen

#-------------------------------------------------------------------------------
  def set_regimens(self, regimens = None):
    self.regimens = regimens
    if self.regimens is None:
      self.regimens = []
    self.n_regimens = len(self.regimens)

#-------------------------------------------------------------------------------
  def add_regimen(self, lrate = None, *lrate_args, **lrate_kwargs):
    self.regimens.append(regimen(self.name + "/regimen_" + str(i), self.dev))
    self.regimens[-1].set_learning_rate(lrate, *lrate_args, **lrate_kwargs)
    self.n_regimens = len(self.regimens)
    return self.set_train_regimen(self.n_regimens - 1)

#-------------------------------------------------------------------------------
  def set_regimen_spec(self, train_regimen = None, dspec = None, pspec = None):
    self.set_train_regimen(train_regimen)
    self.regimen[self.train_regimen].set_dropout_spec(dspec)
    self.regimen[self.train_regimen].set_parameter_spec(pspec)

#-------------------------------------------------------------------------------
  def setup(self, gst = None, ist = None, **kwds):
    if self.trainee is None: return 
    if self.trainee.inputs is None:
      raise AttributeError("Cannot setup trainer without network inputs specified.")

    # Setup the trainer
    self._setup_learning(gst)
    self._setup_optimiser(ist)

    # Then set up the trainee (i.e. the entire network)
    self.trainee.setup(**kwds)

#-------------------------------------------------------------------------------
  def _setup_learning(self, gst = None): # this sets up batch_size, regimen index, and learning_rate
    if self.gst is None: self.set_global_step(gst)
    if self.gst is None: self._setup_global_step()

    # The learning_rate update-op must be done manually during the session
    if self.dev is None:
      self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch_size")
      self.batch_size_op = self.batch_size.assign(Shape(self.trainee.inputs)[0])
      self.regimen_index = Creation('var')(0 , trainable=False, name=self.name+"/regimen_index")
      self.learning_rate = Creation('var')(0., trainable=False, name=self.name+"/learning_rate")
    else:
      with Device(self.dev):
        self.batch_size = Creation('var')(0 , trainable=False, name=self.name+"/batch_size")
        self.batch_size_op = self.batch_size.assign(Shape(self.trainee.inputs)[0])
        self.regimen_index = Creation('var')(0 , trainable=False, name=self.name+"/regimen_index")
        self.learning_rate = Creation('var')(0., trainable=False, name=self.name+"/learning_rate")

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
  @abstractmethod
  def train(self, session = None, *args, **kwds):
    pass

#-------------------------------------------------------------------------------

