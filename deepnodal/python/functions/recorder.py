"""
Recorder module for Tensorflow. It is an abstract class with self.__call__ as the
abstract method which must be defined by inheriting classes for instantiation.

The recorder module abstracts the functionality of collating metrics.
"""

# Gary Bhumbra

#-------------------------------------------------------------------------------
from deepnodal.python.concepts.slave import *
from deepnodal.python.functions.metric import *
import tensorflow as tf

#-------------------------------------------------------------------------------
class recorder (slave):
  """
  A recorder is a slave that record metrics. It is abstract because the absract
  method __call__ is not implemented.
  """
  scalar_objects = None            # list of scalar objects
  scalars = None                   # list of summary scalars
  scalar_labels = None             # list of scalar labels
  scalar_sublabels = None          # list of scalar sublabels
  scalar_metrics = None            # list of metrics
  scalar_accums = None             # list of accumulators
  groups = None                    # spec-keyed dictionary of above

#-------------------------------------------------------------------------------
  def __init__(self, name=None, dev=None):
    slave.__init__(self, name, dev)
    self.set_metrics()

#-------------------------------------------------------------------------------
  def set_metrics(self, metrics=None):
    self.metrics = metrics
    if self.metrics is None:
      self.metrics = []
    self.n_metrics = len(self.metrics)

#-------------------------------------------------------------------------------
  def add_metric(self, creation=None, *args, **_kwds):
    kwds = dict(_kwds)
    name = self.name + "/metrics"
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
    metrics = []
    accums = []
    for metric in self.metrics:
      scalar = metric.ret_scalar(spec)
      if scalar is not None:
        label, sublabel = metric.ret_label(spec)
        scalars.append(scalar)
        objects.append(metric.ret_out())
        labels.append(label)
        sublabels.append(sublabel)
        metrics.append(metric)
        accums.append(metric.ret_accumulator(spec))
    return (objects, scalars, labels, sublabels, metrics, accums)

#-------------------------------------------------------------------------------
  def _call_scalars(self, specs='train'):
    """ Returns a dictionary of appended objects, scalars, labels, sublabels """

    if type(specs) is str: specs = [specs]

    # Initialiase scalar lists
    if self.scalar_objects is None:
      self.scalar_objects = []
    if self.scalars is None:
      self.scalars = []
    if self.scalar_labels is None:
      self.scalar_labels = []
    if self.scalar_sublabels is None:
      self.scalar_sublabels = []
    if self.scalar_metrics is None:
      self.scalar_metrics = []
    if self.scalar_accums is None:
      self.scalar_accums = []

    n = len(self.scalars)

    # Append to scalar lists
    for spec in specs:
      objects, scalars, labels, sublabels, metrics, accums = self.ret_metrics(spec)
      for obj, scalar, label, sublabel, metric, accum in zip(
          objects, scalars, labels, sublabels, metrics, accums):
        self.scalar_objects += [obj]
        self.scalars += [scalar]
        self.scalar_labels += [label]
        self.scalar_sublabels += [sublabel]
        self.scalar_metrics += [metric]
        self.scalar_accums += [accum]

    # Add relevant fields to scalar_dict
    if self.groups is None:
      self.groups = {}

    self.groups.update({spec: {
                               'objects': self.scalar_objects[n:], 
                               'scalars': self.scalars[n:], 
                               'labels': self.scalar_labels[n:], 
                               'sublabels': self.scalar_sublabels[n:],
                               'metrics': self.scalar_metrics[n:],
                               'accums': self.scalar_accums[n:],
                              }})

    return self.groups[spec]

#-------------------------------------------------------------------------------
  def ret_obj_ops_means(self, spec, reset=False):
    group = self.groups[spec]
    obj = []
    ops = []
    means = []
    for i, metric in enumerate(group['metrics']):
      obj.append(group['objects'][i])
      accum = metric.ret_accumulator()
      if isinstance(accum, dict):
        accum = accum[spec]
        if accum is not None:
          ops.append(accum.ret_update_ops(reset=reset))
          means.append(accum.ret_average())
    return obj, ops, means

#-------------------------------------------------------------------------------
  def ret_scalars_ops_means(self, spec, reset=False):
    group = self.groups[spec]
    scalars = group['scalars']
    objects = group['objects']
    means = [None] * len(scalars)
    ops = []
    metrics = group['metrics']
    for i, metric in enumerate(group['metrics']):
      accum = metric.ret_accumulator()
      if isinstance(accum, dict):
        accum = accum[spec]
        if accum is not None:
          ops.append(accum.ret_update_ops(reset=reset))
          means[i] = accum.ret_average()
      if means[i] is None:
        means[i] = objects[i]
    return scalars, ops, means

#-------------------------------------------------------------------------------
  def ret_group(self, spec=None):
    if spec is None:
      return self.groups
    elif spec not in self.groups:
      return None
    return self.groups[spec]

#-------------------------------------------------------------------------------
  def eval_log_scalars(self, session, spec, feed_dict, flush=False, skip_log=False):
    sublabels = self.groups[spec]['sublabels']
    scalars, ops, means = self.ret_scalars_ops_means(spec, reset=True)
    num_scalars = len(scalars)
    scalars_log, scalars_obj = None, None
    if len(ops):
      ops = session.run(ops, feed_dict=feed_dict)
    logs_means = session.run(scalars + means, feed_dict=feed_dict)
    scalars_log = logs_means[:num_scalars]
    scalars_obj = logs_means[num_scalars:(2*num_scalars)]
    if not skip_log:
      self._add_logs(scalars_log, flush=flush)
    scalars_str = [sublbl + "=" + str(obj) for sublbl, obj in zip(
                   sublabels, scalars_obj)]
    return ', '.join(scalars_str)

#-------------------------------------------------------------------------------
