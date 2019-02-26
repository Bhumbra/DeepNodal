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
CIFAR10_TRAIN_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
CIFAR10_TEST_FILES = ['test_batch']
CIFAR10_KEYS = [b'data', b'labels']
CIFAR10_DIMS = [3, 32, 32]
CIFAR100_TRAIN_FILES = ['train']
CIFAR100_TEST_FILES = ['test']
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
  def __init__(self, directory = CIFAR_DIRECTORY, 
                     train_files = CIFAR10_TRAIN_FILES, 
                     test_files = CIFAR10_TEST_FILES, 
                     keys = CIFAR10_KEYS, 
                     dims = CIFAR10_DIMS,
                     depth_to_last_dim = True, border_val = 0):
    maybe_download_and_extract(directory, CIFAR10_SOURCE)
    if type(CIFAR10_SOURCE) is dict:
      directory += CIFAR10_SOURCE[list(CIFAR10_SOURCE)[0]] + '/'
    else:
      directory += CIFAR10_SOURCE
    imager.__init__(self, directory, train_files, test_files, keys, dims, 
                    depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
class cifar100 (imager):
#-------------------------------------------------------------------------------
  def __init__(self, directory = CIFAR_DIRECTORY, 
                     train_files = CIFAR100_TRAIN_FILES, 
                     test_files = CIFAR100_TEST_FILES, 
                     keys = CIFAR100_KEYS, 
                     dims = CIFAR100_DIMS,
                     depth_to_last_dim = True, border_val = 0):
    maybe_download_and_extract(directory, CIFAR100_SOURCE)
    if type(CIFAR100_SOURCE) is dict:
      directory += CIFAR100_SOURCE[list(CIFAR100_SOURCE)[0]] + '/'
    else:
      directory += CIFAR100_SOURCE
    if directory[-1] != '/':
      directory += '/'
    imager.__init__(self, directory, train_files, test_files, keys, dims, 
                    depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------

