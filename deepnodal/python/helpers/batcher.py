"""
Class to batch data conveniently
"""

#-------------------------------------------------------------------------------
import pickle
import collections
import numpy as np

#-------------------------------------------------------------------------------
DEFAULT_SET_NAME  ='train'
DEFAULT_SETS = {0: ['train'],
                1: ['train'],
                2: ['train', 'test'],
                3: ['train', 'eval', 'test']}
                
#-------------------------------------------------------------------------------
class batcher (object):
  # Public
  directory = None
  files = None
  sets = None

  # Private
  _inputs = None
  _labels = None
  _counts = None

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], set_specs=[], files=[], directory=''):
    self.add_sets(set_names, set_specs)
    self.set_files(files, directory)

#-------------------------------------------------------------------------------
  def add_set(self, set_name=None, set_spec=None):
    if set_name is None:
      return self.sets
    if self.sets is None:
      self.sets = collections.OrderedDict()
    self.sets.update({set_name: {'spec': set_spec,
                                 'indices': None,
                                 'support': None,
                                 'inputs': None,
                                 'labels': None,
                                 'counter': None,
                                 'arange': None,
                                 'permute': None}})

#-------------------------------------------------------------------------------
  def add_sets(self, set_names=[], set_specs=[]):
    return [self.add_set(set_name, set_spec) for set_name, set_spec in
                                             zip(set_names, set_specs)]

#-------------------------------------------------------------------------------
  def set_files(self, files=[], directory=''):
    if type(files) is str: files = [files]
    self.files = files
    self.directory = directory

#-------------------------------------------------------------------------------
  def read_data(self, *args):
    """
    A Pickle reader, with no relationship to partitioning.
    """
    input_key, label_key = args
    if self.directory is None: raise ValueError("Directory not set")
    if self.files is None: raise ValueError("Data files not set")
    inputs = []
    labels = []
    counts = []
    for data_file in self.files:
      with open(self.directory + data_file, 'rb') as open_file:
        data_dict = pickle.load(open_file, encoding = 'bytes')
      inputs.append(data_dict[input_key])
      labels.append(data_dict[label_key])
      counts.append(len(labels[-1]))
    self._counts = counts
    return self.set_data(np.concatenate(inputs), 
                         np.concatenate(labels))

#-------------------------------------------------------------------------------
  def set_data(self, inputs, labels=None, set_name=None):
    if set_name:
      self.sets[set_name].update({'inputs': inputs, 'labels': labels}) 
      return self.sets[set_name]
    self._inputs = inputs
    self._labels = labels
    return self._inputs, self._labels

#-------------------------------------------------------------------------------
  def partition(self, set_names=[], set_specs=[], randomise=True, seed=None):
    if seed is not None:
      np.random.seed(seed)
    if set_names:
      if self._sets:
        raise AttributeError("Data set specifications already set.")
      else:
        self.add_sets(set_names, set_specs)
    elif self.sets is None:
      num_counts = len(self._counts)
      if num_counts < 1:
        set_names = DEFAULT_SETS[0]
        set_specs = [1.]
      elif num_counts in DEFAULT_SETS:
        set_names = DEFAULT_SETS[num_counts]
        set_specs = self._counts
      else:
        counts = [sum(self._counts[:-1]), self._counts[-1]]
        set_names = DEFAULT_SETS[len(counts)]
        set_specs = counts
      self.add_sets(set_names, set_specs)
    elif self._inputs is None:
      raise AttributeError("Must invokve set_data first")

    set_names = list(self.sets.keys())
    set_specs = [self.sets[set_name]['spec'] for set_name in set_names]
    
    # Check spec type
    spec_type = None
    for spec in set_specs:
      if spec_type is None:
        spec_type = type(spec)
      elif type(spec) is not spec_type:
        raise TypeError("Inconsistent specification types: {} vs {}".
                        format(type(spec), spec_type))

    if not randomise:
      indices = np.arange(len(self._inputs), dtype = int)
    else:
      indices = np.random.permutation(len(self._inputs))

    # Integer spec
    if spec_type is int:
      cumsum_spec = np.hstack([0, np.cumsum(set_specs)])
      mod_indices = np.mod(indices, cumsum_spec[-1])
      for i in range(len(set_specs)):
        self.sets[set_names[i]]['indices'] = indices[
            np.logical_and(mod_indices >= cumsum_spec[i],
                           mod_indices <  cumsum_spec[i+1])]

    # Float case
    elif spec_type is float:
      intervals = np.round(np.atleast_1d(spec) * float(len(self._inputs)))
      start = 0
      for i in range(len(set_specs)):
        finish = start + int(intervals[i])
        self.sets[set_name[i]]['indices'] = indices[start:finish]
        start = finish

    # Explicit case
    else:
      if randomise:
        raise ValueError("Cannot randomise explicit set specifications")
      for i in range(len(set_specs)):
        self.sets[set_name[i]]['indices'] = self.sets[set_name[i]]['spec']

    # Assign data
    for key, val in self.sets.items():
      indices = val['indices']
      self.sets[key]['inputs'] = [self._inputs[i] for i in indices]
      if self._labels is not None:
        self.sets[key]['labels'] = [self._labels[i] for i in indices]
      self.sets[key]['support'] = len(indices)
      self.sets[key]['arange'] = np.arange(self.sets[key]['support'], 
                                   dtype=int)
    return self.sets

#-------------------------------------------------------------------------------
  def _repermute(self, set_name, randomise=True):
    if not randomise:
      self.sets[set_name]['permute'] = self.sets[set_name]['arange']
    else:
      self.sets[set_name]['permute'] = np.random.permutation(
                                         self.sets[set_name]['support'])
    self.sets[set_name]['counter'] = 0

#-------------------------------------------------------------------------------
  def next_batch(self, set_name=DEFAULT_SET_NAME, batch_size=None, randomise=None):

    # Set defaults
    multiset = isinstance(set_name, (list, tuple))
    if batch_size is None and not multiset:
      if randomise is None: 
        randomise = False
      batch_size = self.sets[set_name]['support']
      self.sets[set_name]['counter'] = None
    elif randomise is None:
      randomise = True

    # Handle multiset case first
    if multiset:
      num_sets = len(set_name)
      if batch_size % num_sets:
        raise ValueError("Batch size indivisible by number of data sets")
      subbatch_size = batch_size / num_sets
      data = [None] * len(set_name)
      reset = False
      for i, name in enumerate(set_name):
        data[i] = self.next_batch(name, subbatch_size, randomise)
        reset = data[i] is None
        if reset:
          break
      if reset:
        for name in set_name:
          self.sets[name]['counter'] = None
        return None
      else:
        inputs, labels = zip(*data)
        if labels[0] is None:
          return np.concatenate(images, axis=0)
        return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)
      
    # Now single set
    counter =  self.sets[set_name]['counter']
    if counter is None:
      self._repermute(set_name, randomise)
      counter = self.sets[set_name]['counter']
    elif counter + batch_size > self.sets[set_name]['support']:
      self.sets[set_name]['counter'] = None
      return None

    # Permute single set
    new_counter = counter + batch_size
    self.sets[set_name]['counter'] = new_counter
    idx = self.sets[set_name]['permute'][counter:new_counter]
    inputs, labels = self.sets[set_name]['inputs'], self.sets[set_name]['labels']
    if type(inputs) is np.ndarray:
      batch_inputs = inputs[idx]
    else:
      batch_inputs = np.array([inputs[i] for i in idx])
    if labels is None:
      return batch_inputs
    if type(labels) is np.ndarray:
      batch_labels = labels[idx]
    else:
      batch_labels = np.array([labels[i] for i in idx])
    return batch_inputs, batch_labels

#-------------------------------------------------------------------------------
