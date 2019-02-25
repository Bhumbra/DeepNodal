"""
Recorder module for Tensorflow. It is an abstract class with self.__call__ as the
abstract method which must be defined by inheriting classes for instantiation.

The recorder module abstracts the functionality of collating metrics.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.slave import *
from deepnodal.python.functions.metric import *

#-------------------------------------------------------------------------------
class recorder (slave):
  """
  A recorder is a slave that record metrics. It is abstract because the absract
  method __call__ is not implemented.
  """
  scalars = None                   # list of summary scalars
  scalar_labels = None             # list of scalar labels
  scalar_sublabels = None          # list of scalar sublabels
  scalar_logs = None               # list of scalar logs
  scalar_group = None              # spec-keyed dictionary of above

#-------------------------------------------------------------------------------
  def __init__(self, name = None, dev = None):
    slave.__init__(self, name, dev)
    self.set_metrics()

#-------------------------------------------------------------------------------
  def set_metrics(self, metrics = None):
    self.metrics = metrics
    if self.metrics is None:
      self.metrics = []
    self.n_metrics = len(self.metrics)

#-------------------------------------------------------------------------------
  def add_metric(self, creation = None, *args, **_kwds):
    kwds = dict(_kwds)
    name = self.name + "/metrics/metric_{}".format(self.n_metrics)
    dev = self.dev
    if 'name' in kwds:
      name = kwds['name']
    if 'dev' in kwds:
      dev = kwds['dev']
      kwds.pop('dev')
    dev = None # metrics cannot be assigned to GPUs
    self.metrics.append(metric(name, dev))
    self.n_metrics = len(self.metrics)
    self.metrics[-1].set_creation(creation, *args, **kwds)
    return self.metrics[-1]

#-------------------------------------------------------------------------------
  def ret_metrics(self, spec=None):
    """
    Returns self.metrics is spec is None otherwise returns a tuple of
    (scalars, objects, labels, sublabels). 
    
    Typically spec is None, 'train', or 'test'.
    """
    if spec is None: return self.metrics
    scalars = []
    objects = []
    labels = []
    sublabels = []
    for metric in self.metrics:
      scalar = metric.ret_scalar(spec)
      if scalar is not None:
        label, sublabel = metric.ret_label(spec)
        scalars.append(scalar)
        objects.append(metric.ret_out())
        labels.append(label)
        sublabels.append(sublabel)
    return (objects, scalars, labels, sublabels)

#-------------------------------------------------------------------------------
  def _call_scalars(self, specs = 'train'):
    """ Returns appended scalars, labels, sublabels, and_logs """

    if type(specs) is str: specs = [specs]

    # Initialiase scalar lists
    if self.scalars is None:
      self.scalars = []
    if self.scalar_labels is None:
      self.scalar_labels = []
    if self.scalar_sublabels is None:
      self.scalar_sublabels = []
    if self.scalar_logs is None:
      self.scalar_logs = []

    n = len(self.scalars)

    # Append to scalar lists
    for spec in specs:
      scalars, objects, labels, sublabels = self.ret_metrics(spec)
      for scalar, _, label, sublabel in zip(
          scalars, objects, labels, sublabels):
        self.scalars += [scalar]
        self.scalar_labels += [label]
        self.scalar_sublabels += [sublabel]
        self.scalar_logs += [Summary('scalar')(label, scalar)]

    # Add relevant fields to scalar_dict
    if self.scalar_group is None:
      self.scalar_group = {}

    self.scalar_group.update({spec: [self.scalars[n:], self.scalar_labels[n:],
           self.scalar_sublabels[n:], self.scalar_logs[n:]]})

    return self.scalar_group[spec]

#-------------------------------------------------------------------------------
  def ret_scalar_group(self, spec = 'train'):
    if spec not in self.scalar_group:
      return None
    return self.scalar_group[spec]

#-------------------------------------------------------------------------------
