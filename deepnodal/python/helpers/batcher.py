"""
Class to batch data conveniently
"""

#-------------------------------------------------------------------------------
import numpy as np
import collections

#-------------------------------------------------------------------------------
DEFAULT_SET_NAME = 'train'

#-------------------------------------------------------------------------------
class Batcher (object):
  # Public
  sets = None

  # Private
  _inputs = None
  _labels = None

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], set_specs=[]):
    self.add_sets(set_names, set_specs)

#-------------------------------------------------------------------------------
  def add_set(self, set_name=None, set_spec=None):
    if set_name is None:
      return self.sets
    if self.sets is None:
      self.sets = collections.OrderedDict()
    self.set_sets(self.sets.update({set_name: {'spec': set_spec,
                                               'indices': None,
                                               'inputs': None,
                                               'labels': None,
                                               'counter': None,
                                               'permute': None}}))

#-------------------------------------------------------------------------------
  def add_sets(self, set_names=[], set_specs=[]):
    return [add_set(set_name, set_spec) for set_name, set_spec in
                                       zip(set_names, set_specs)]

#-------------------------------------------------------------------------------
  def set_data(self, inputs, labels = None, set_name = None):
    if set_name:
      self.sets[set_name].update({'inputs': inputs, 'labels': labels}) 
      return self.sets[set_name]
    self._inputs = inputs
    self._labels = labels

#-------------------------------------------------------------------------------
  def partition(self, randomise=True):
    if self._inputs is None:
      raise AttributeError("Must invokve set_data first")
    if self.sets is None:
      self.add_spec(DEFAULT_SET_NAME, 1.)

    set_names = list(self.sets.keys())
    set_specs = [self.sets[set_name]['spec'] for set_name in set_names]
    
    # Check spec type
    spec_type = None
    for spec in set_specs:
      if spec_type is None:
        spec_type = type(spec)
      elif type(spec) is not spec_type:
        raise TypeError("Inconsistent specification types")

     
    if not randomise:
      indices = np.arange(len(self._inputs), dtype = int)
    else:
      indices = np.random.permutation(len(self._inputs))

    # Integer spec
    if spec_type is int:
      cumsum_spec = np.hstack([0, np.cumsum(set_specs)])
      mod_indices = np.mod(indices, cumsum_spec[-1])
      for i in range(len(set_specs)):
        self.sets[set_name[i]]['indices'] = indices[
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
      self.sets[key]['inputs'] = [self._inputs[i] for i in val['indices']]
      if self._labels:
        self.sets[key]['labels'] = [self._labels[i] for i in val['indices']]

    return self.sets

#-------------------------------------------------------------------------------
  def _repermute(self, set_name, randomise=True):
    n = len(self.sets[set_name]['inputs'])

    if not randomise:
      self.sets[set_name]['permute'] = np.arange(n, dtype = int)
    else:
      self.sets[set_name]['permute'] = np.random.permutation(n)

#-------------------------------------------------------------------------------
  def next_batch(self, set_name=DEFAULT_SET_NAME, batch_size=1, randomise=True):
    counter =  self.sets[set_name]['counter']
    if counter is None:
      self._repermute(set_name, randomise)

    elif counter + batch_size > len(self.sets[set_name]['inputs']):
      self.sets[set_name]['counter'] = None
      return None

    new_counter = counter + batch_size
    idx = np.arange(counter, new_counter)
    self.sets[set_name]['counter'] = new_counter
    inputs, labels = self.sets.set_name['inputs'], self.sets.set_name['babels']
    batch_inputs = [inputs[i] for i in idx]
    if labels is None:
      return batch_inputs
    batch_labels = [labels[i] for i in idx]
    return batch_inputs, batch_labels

#-------------------------------------------------------------------------------
