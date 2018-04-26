from deepnodal.python.helpers.imager import *

#-------------------------------------------------------------------------------
CIFAR10_DIRECTORY = "/tmp/cifar-10-batches-py/"
CIFAR10_TRAIN_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
CIFAR10_TEST_FILES = ['test_batch']
CIFAR10_KEYS = [b'data', b'labels']
CIFAR10_DIMS = [3, 32, 32]

#-------------------------------------------------------------------------------
CIFAR100_DIRECTORY = "/tmp/cifar-100-python/"
CIFAR100_TRAIN_FILES = ['train']
CIFAR100_TEST_FILES = ['test']
CIFAR100_KEYS = [b'data', b'fine_labels']
CIFAR100_DIMS = [3, 32, 32]

#-------------------------------------------------------------------------------
class cifar10 (imager):
#-------------------------------------------------------------------------------
  def __init__(self, directory = CIFAR10_DIRECTORY, 
                     train_files = CIFAR10_TRAIN_FILES, 
                     test_files = CIFAR10_TEST_FILES, 
                     keys = CIFAR10_KEYS, 
                     dims = CIFAR10_DIMS,
                     depth_to_last_dim = True, border_val = 0):
    imager.__init__(self, directory, train_files, test_files, keys, dims, 
                    depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
class cifar100 (imager):
#-------------------------------------------------------------------------------
  def __init__(self, directory = CIFAR100_DIRECTORY, 
                     train_files = CIFAR100_TRAIN_FILES, 
                     test_files = CIFAR100_TEST_FILES, 
                     keys = CIFAR100_KEYS, 
                     dims = CIFAR100_DIMS,
                     depth_to_last_dim = True, border_val = 0):
    imager.__init__(self, directory, train_files, test_files, keys, dims, 
                    depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------

