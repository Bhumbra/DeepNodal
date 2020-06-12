from deepnodal.python.helpers.imager import *
import os
import urllib
import numpy as np
from scipy.io import loadmat
from deepnodal.python.cloud.gs import *

#-------------------------------------------------------------------------------
SVHN_DIRECTORY = '/tmp/'
SVHN_URL_DIRECTORY = 'http://ufldl.stanford.edu/housenumbers/'
SVHN_FILES = ['train_32x32.mat', 'extra_32x32.mat', 'test_32x32.mat'] 
SVHN_KEYS = ('X', 'y')
SVHN_DIMS = [32, 32, 3]
SVHN_DEFAULT_SETS = {0: ['train'],         # To force including extra in train
                     1: ['train'],
                     2: ['train', 'test']}

#-------------------------------------------------------------------------------
def maybe_download(directory, file_name):
  source_path = SVHN_URL_DIRECTORY + file_name
  target_path = os.path.join(directory, file_name)
  if not os.path.exists(target_path):
    print("Downloading %s to %s" % (source_path, directory))
    urllib.request.urlretrieve(source_path, target_path)

#-------------------------------------------------------------------------------
class SVHN (imager):

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], 
                     set_spec=[],
                     files=SVHN_FILES, 
                     directory=SVHN_DIRECTORY, 
                     dims=SVHN_DIMS,
                     depth_last_dim=False): # it's actually True but no swapping is needed
    if directory[:5] != 'gs://':
      for file_name in files:
        maybe_download(directory, file_name)
    super().__init__(set_names, set_spec, files, directory,
                    dims, depth_last_dim)

#-------------------------------------------------------------------------------
  def read_data(self, *args, gcn=False, zca=False, gcn_within_depth=True):
    if self.dims is None: raise ValueError("Data dimensions not set")
    inputs = []
    labels = []
    counts = []
    for data_file in self.files:
      data_path = os.path.join(self.directory, data_file)
      mat_data = loadmat(data_path)
      X = np.moveaxis(mat_data['X'], [0, 1, 2, 3], [1, 2, 3, 0]).copy(order='C')
      y = np.ravel(np.array(mat_data['y'], dtype=int)).copy(order='C')
      inputs.append(X / 255)
      labels.append(y % 10)
      counts.append(len(labels[-1]))
    self._counts = counts
    return super().read_data(np.concatenate(inputs, axis=0), 
                             np.concatenate(labels), gcn=gcn, zca=zca, 
                             gcn_within_depth=gcn_within_depth)

#-------------------------------------------------------------------------------
  def partition(self, set_names=[], set_specs=[], randomise=True, seed=None, 
      default_sets=SVHN_DEFAULT_SETS):
    return super().partition(set_names, set_specs, randomise, seed, default_sets)

#-------------------------------------------------------------------------------
