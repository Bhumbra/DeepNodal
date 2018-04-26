from deepnodal.python.helpers.imager import *
import os
import urllib
import struct
import subprocess
import numpy as np

#-------------------------------------------------------------------------------
MNIST_URL_DIRECTORY = 'http://yann.lecun.com/exdb/mnist/'
MNIST_ZIP_EXTENSION = '.gz'
MNIST_DIRECTORY = "/tmp/mnist/"
MNIST_TRAIN_FILES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
MNIST_TEST_FILES = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
MNIST_KEYS = None
MNIST_DIMS = [1, 28, 28]
MNIST_UINT8_DTYPE = np.dtype(np.uint8).newbyteorder('>')
MNIST_UINT32_DTYPE = np.dtype(np.uint32).newbyteorder('>')

#-------------------------------------------------------------------------------
def maybe_download_and_extract(directory, filename, zipextension=MNIST_ZIP_EXTENSION):
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
    gunzip_command = "gunzip " + source_path
    gunzip_process = subprocess.Popen(gunzip_command.split(), stdout=subprocess.PIPE)
    gunzip_process.communicate()

#-------------------------------------------------------------------------------
def read_idx1(file_path):
  with open(file_path, 'rb') as f:
    magic, count = struct.unpack('>2I', f.read(8))
    data = np.fromstring(f.read(), dtype=MNIST_UINT8_DTYPE)
  return data

#-------------------------------------------------------------------------------
def read_idx3(file_path):
  with open(file_path, 'rb') as f:
    magic, count, rows, cols = struct.unpack('>4I', f.read(16))
    data = np.fromstring(f.read(), dtype=MNIST_UINT8_DTYPE)
  return data.reshape([count, rows, cols])

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class mnist (imager):
#-------------------------------------------------------------------------------
  def __init__(self, directory = MNIST_DIRECTORY, 
                     train_files = MNIST_TRAIN_FILES, 
                     test_files = MNIST_TEST_FILES, 
                     keys = MNIST_KEYS, 
                     dims = MNIST_DIMS,
                     depth_to_last_dim = True, border_val = 0):
    for filename in train_files:
      maybe_download_and_extract(directory, filename)
    for filename in test_files:
      maybe_download_and_extract(directory, filename)
    imager.__init__(self, directory, train_files, test_files, keys, dims, 
                    depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
  def read_data(self, gcn = False, zca = False, gcn_within_depth = True):
    # MNIST source data is not conveniently pickled.
    train_images, train_labels = [], []
    self.train_num_examples = 0
    for data_file in self.train_files:
      if data_file.find('idx3') >= 0:
        train_images.append(read_idx3(self.directory + data_file))
        self.train_num_examples += len(train_images[-1])
      else:
        train_labels.append(read_idx1(self.directory + data_file))

    test_images, test_labels = [], []
    self.test_num_examples = 0
    for data_file in self.test_files:
      if data_file.find('idx3') >= 0:
        test_images.append(read_idx3(self.directory + data_file))
        self.test_num_examples += len(test_images[-1])
      else:
        test_labels.append(read_idx1(self.directory + data_file))

    train_dims = np.hstack((self.train_num_examples, self.dims))
    self.train_images = np.vstack(train_images).reshape(train_dims) / 255.
    self.train_labels = np.hstack(train_labels)
    test_dims = np.hstack((self.test_num_examples, self.dims))
    self.test_images = np.vstack(test_images).reshape(test_dims) / 255.
    self.test_labels = np.hstack(test_labels)

    depth_axis = 1 if gcn_within_depth else None
    if self.gcn: # default axes should be fine
      self.train_images = global_contrast_norm(self.train_images, depth_axis = depth_axis)
      self.test_images = global_contrast_norm(self.test_images, depth_axis = depth_axis)

    if self.zca: 
      print("Warning: performing ZCA-whitening of single-channel data!")
      self.train_images = zca_whitening(self.train_images)
      self.test_images = zca_whitening(self.test_images)

    self.batch_index = self.train_num_examples # forces a repermutation on first update

    if self.depth_to_last_dim:
      train_dims = np.hstack((train_dims[0], train_dims[2:], train_dims[1]))
      self.train_images = np.swapaxes(np.swapaxes(self.train_images, 1, 3), 1, 2).reshape(train_dims)
      test_dims = np.hstack((test_dims[0], test_dims[2:], test_dims[1]))
      self.test_images = np.swapaxes(np.swapaxes(self.test_images, 1, 3), 1, 2).reshape(test_dims)

#-------------------------------------------------------------------------------
