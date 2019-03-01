from deepnodal.python.helpers.imager import *
import os
import urllib
import struct
import subprocess
import numpy as np
import tarfile

#-------------------------------------------------------------------------------
CIFAR_DIRECTORY = '/tmp/'
CIFAR_URL_DIRECTORY = 'https://www.cs.toronto.edu/~kriz/'
CIFAR_ZIP_EXTENSION = '.tar.gz'
CIFAR_GZIP_COMMAND = 'tar fxz'
CIFAR10_SOURCE = {'cifar-10-python':'cifar-10-batches-py'}
CIFAR100_SOURCE = 'cifar-100-python'
CIFAR10_FILES = ['data_batch_1', 'data_batch_2', 
                 'data_batch_3', 'data_batch_4', 
                 'data_batch_5', 'test_batch']
CIFAR10_KEYS = [b'data', b'labels']
CIFAR10_DIMS = [3, 32, 32]
CIFAR100_FILES = ['train', 'test']
CIFAR100_KEYS = [b'data', b'fine_labels']
CIFAR100_DIMS = [3, 32, 32]

#-------------------------------------------------------------------------------
def maybe_download_and_extract(directory, _subdir, zipextension=CIFAR_ZIP_EXTENSION):
  subdir = list(_subdir)[0] if type(_subdir) is dict else _subdir
  if os.path.exists(directory + subdir): return
  source_path = CIFAR_URL_DIRECTORY + subdir + zipextension
  target_path = directory + subdir + zipextension
  if not(os.path.exists(target_path)):
    print("Downloading %s from %s" % (subdir+zipextension, CIFAR_URL_DIRECTORY))
    urllib.request.urlretrieve(source_path, target_path)
  subdir = subdir if type(_subdir) is str else _subdir[subdir]
  source_path = target_path
  target_path = directory + subdir
  if not(os.path.exists(target_path)):
    print("Extracting %s" % (source_path))
    tar = tarfile.open(source_path, 'r:gz')
    # Can't make python's gzip work here so using gunzip
    gzip_command = CIFAR_GZIP_COMMAND + " " + source_path + " -C " + CIFAR_DIRECTORY
    gzip_process = subprocess.Popen(gzip_command.split(), stdout=subprocess.PIPE)
    gzip_process.communicate()

#-------------------------------------------------------------------------------
class cifar10 (imager):

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], 
                     set_spec=[],
                     files=CIFAR10_FILES, 
                     directory=CIFAR_DIRECTORY, 
                     dims=CIFAR10_DIMS,
                     depth_to_last_dim=True, 
                     border_val = 0):
    maybe_download_and_extract(directory, CIFAR10_SOURCE)
    if type(CIFAR10_SOURCE) is dict:
      directory += CIFAR10_SOURCE[list(CIFAR10_SOURCE)[0]] + '/'
    else:
      directory += CIFAR10_SOURCE
    if directory[-1] != '/':
      directory += '/'
    super().__init__(set_names, set_spec, files, directory,
                    dims, depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
  def read_data(self, input_spec=CIFAR10_KEYS[0], label_spec=CIFAR10_KEYS[1], 
                      gcn=False, zca=False, gcn_within_depth=True):
    return super().read_data(input_spec, label_spec, gcn, zca, gcn_within_depth)

#-------------------------------------------------------------------------------
class cifar100 (imager):

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], 
                     set_spec=[],
                     files=CIFAR100_FILES, 
                     directory=CIFAR_DIRECTORY, 
                     dims=CIFAR100_DIMS,
                     depth_to_last_dim=True, 
                     border_val = 0):
    maybe_download_and_extract(directory, CIFAR10_SOURCE)
    if type(CIFAR10_SOURCE) is dict:
      directory += CIFAR10_SOURCE[list(CIFAR10_SOURCE)[0]] + '/'
    else:
      directory += CIFAR10_SOURCE
    if directory[-1] != '/':
      directory += '/'
    super().__init__(set_names, set_spec, files, directory,
                    dims, depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
  def read_data(self, input_spec=CIFAR100_KEYS[0], label_spec=CIFAR100_KEYS[1], 
                      gcn=False, zca=False, gcn_within_depth=True):
    return super()(input_spec, label_spec, gcn, zcn, gcn_within_depth)

#-------------------------------------------------------------------------------
