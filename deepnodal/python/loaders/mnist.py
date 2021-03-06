from deepnodal.python.helpers.imager import *
import os
import urllib
import struct
import subprocess
import numpy as np
from deepnodal.python.cloud.gs import *

#-------------------------------------------------------------------------------
MNIST_GZIP_COMMAND = "gunzip"
MNIST_URL_DIRECTORY = 'http://yann.lecun.com/exdb/mnist/'
MNIST_ZIP_EXTENSION = '.gz'
MNIST_FILES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
               't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
MNIST_DIRECTORY = "/tmp/mnist/"
MNIST_KEYS = None
MNIST_DIMS = [1, 28, 28]
MNIST_UINT8_DTYPE = np.dtype(np.uint8).newbyteorder('>')
MNIST_UINT32_DTYPE = np.dtype(np.uint32).newbyteorder('>')

#-------------------------------------------------------------------------------
def maybe_download_and_extract(filename, directory, zipextension=MNIST_ZIP_EXTENSION):
  if os.path.exists(directory + filename): return
  source_path = MNIST_URL_DIRECTORY + filename + zipextension
  target_path = directory + filename + zipextension
  if not(os.path.exists(target_path)):
    if not(os.path.exists(directory)):
      os.makedirs(directory)
    print("Downloading %s from %s" % (filename+zipextension, MNIST_URL_DIRECTORY))
    urllib.request.urlretrieve(source_path, target_path)
  source_path = target_path
  target_path = directory + filename
  if not(os.path.exists(target_path)):
    print("Extracting %s" % (source_path))
    # Can't make python's gzip work here so using gunzip
    gzip_command = MNIST_GZIP_COMMAND + " " + source_path
    gzip_process = subprocess.Popen(gzip_command.split(), stdout=subprocess.PIPE)
    gzip_process.communicate()


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class mnist (imager):
  read_gcs = None

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], 
                     set_spec=[],
                     files=MNIST_FILES, 
                     directory=MNIST_DIRECTORY, 
                     dims=MNIST_DIMS,
                     depth_last_dim=True):
    if directory[:5] == 'gs://':
      self.read_gcs = GCS(directory)
    else:
      for data_file in files:
        maybe_download_and_extract(data_file, directory)
    super().__init__(set_names, set_spec, files, directory,
                    dims, depth_last_dim)

#-------------------------------------------------------------------------------
  def read_data(self, *args, 
                gcn=False, zca=False, gcn_within_depth=True):
    # MNIST source data is not conveniently pickled so we overload here.
    if self.directory is None: raise ValueError("Directory not set")
    if self.files is None: raise ValueError("Data files not set")

    def _read_idx1(file_path):
      with open(file_path, 'rb') as bin_read:
        magic, count = struct.unpack('>2I', bin_read.read(8))
        data = np.fromstring(bin_read.read(), dtype=MNIST_UINT8_DTYPE)
      return data

    def _read_idx3(file_path):
      with open(file_path, 'rb') as bin_read:
        magic, count, rows, cols = struct.unpack('>4I', bin_read.read(16))
        data = np.fromstring(bin_read.read(), dtype=MNIST_UINT8_DTYPE)
      return data.reshape([count, rows, cols])

    inputs = []
    labels = []
    counts = []
    for data_file in self.files:
      data_path = self.directory + data_file
      if data_file.find('idx3') >= 0:
        if self.read_gcs:
          data = self.read_gcs(_read_idx3, data_path, 'read')
        else:
          data = _read_idx3(data_path)
        inputs.append(data)
      else:
        if self.read_gcs:
          data = self.read_gcs(_read_idx1, data_path, 'read')
        else:
          data = _read_idx1(data_path)
        labels.append(data)
        counts.append(len(labels[-1]))
    self._counts = counts
    return super().read_data(np.concatenate(inputs, axis=0)/255., 
                             np.concatenate(labels), gcn=gcn, zca=zca, 
                             gcn_within_depth=gcn_within_depth)

#-------------------------------------------------------------------------------
