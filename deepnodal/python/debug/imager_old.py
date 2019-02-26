# A generic image handler for machine learning

# Gary Bhumbra.

#--------------------------------------------------------------------------------
from deepnodal.python.helpers.scalers import *
import numpy as np
import pickle

#--------------------------------------------------------------------------------
def global_middle_scale(data, middle = True, scale = True, batch_axis = None, depth_axis = None, 
    locat_val = 0., scale_val = 2., dtype = float):
  # This is just a fancy name for middle-scale() except with pre-ordained batch and depth axes
  tr_axes = np.arange(data.ndim)
  tr_axes = tr_axes[tr_axes != batch_axis]
  if depth_axis is not None:
    if depth_axis < 0: depth_axis += data.ndim
    tr_axes = tr_axes[tr_axes != depth_axis] 
  return middle_scale(data, tr_axes, tr_axes, dtype = dtype, locat_val = locat_val, scale_val = scale_val)


#--------------------------------------------------------------------------------
def global_contrast_norm(data, center = True, scale = True, batch_axis = 0, depth_axis = 1,
    locat_val = 0., scale_val = 2., dtype = float):
  # This is just a fancy name for center-scale() except with pre-ordained batch and depth axes
  tr_axes = np.arange(data.ndim)
  tr_axes = tr_axes[tr_axes != batch_axis]
  if depth_axis is not None:
    if depth_axis < 0: depth_axis += data.ndim
    tr_axes = tr_axes[tr_axes != depth_axis] 
  return center_scale(data, tr_axes, tr_axes, dtype = dtype, locat_val = locat_val, scale_val = scale_val)

#--------------------------------------------------------------------------------
def zca_whitening(_data, batch_axis = 0, depth_axis = 1, dtype = float):
  data = _data.astype(dtype)
  dims = np.atleast_1d(data.shape)
  if depth_axis < 0: depth_axis += len(dims)
  slices = [slice(None)] * len(dims)
  n_batch = dims[batch_axis]
  bd_axes = [batch_axis, depth_axis]
  tr_axes = []
  for i in range(len(dims)):
    if i not in bd_axes:
      tr_axes.append(i)
  tr_axes = np.array(tr_axes)
  tr_dims = dims[tr_axes]

  # Numpy defaults to row-major 
  n_rows = dims[depth_axis]
  n_cols = np.prod(tr_dims)
  mat = np.empty([n_rows, n_cols], dtype = dtype)
  for i in range(n_batch):
    slices[batch_axis] = i
    for j in range(n_rows):
      slices[depth_axis] = j
      mat[j, :] = np.ravel(data[tuple(slices)])
    cov = np.cov(mat)
    U, S, V = np.linalg.svd(cov)
    zca = np.dot(U, np.dot(np.diag(1./np.sqrt(S + FLOAT_EPSILON)), U.T))
    mat = np.dot(zca, mat)
    for j in range(n_rows):
      slices[depth_axis] = j
      data[tuple(slices)] = mat[j, :].reshape(tr_dims)

  return data

#-------------------------------------------------------------------------------
class imager (object):
  directory = None
  train_files = None
  test_files = None
  keys = None
  dims = None
  depth_to_last_dim = None
  border_val = None
  train_num_examples = None
  test_num_examples = None
  gcn = None
  zca = None
  train_permute = None
  batch_index = None

#-------------------------------------------------------------------------------
  def __init__(self, directory = None, train_files = None, test_files = None, 
               keys = None, dims = None, depth_to_last_dim = True, border_val = 0):
    self.set_directory(directory)
    self.set_files(train_files, test_files)
    self.set_keys(keys)
    self.set_dims(dims, depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
  def set_keys(self, keys = None):
    self.keys = keys # in the form [data_key, labels_key]
    
#-------------------------------------------------------------------------------
  def set_dims(self, dims = None, depth_to_last_dim = False, border_val = 0):
    self.dims = dims
    self.depth_to_last_dim = depth_to_last_dim
    self.border_val = border_val

#-------------------------------------------------------------------------------
  def set_directory(self, directory = None):
    self.directory = directory

#-------------------------------------------------------------------------------
  def set_files(self, train_files = None, test_files = None):
    self.train_files = train_files
    self.test_files = test_files

#-------------------------------------------------------------------------------
  def read_data(self, gcn = False, zca = False, gcn_within_depth = True):
    self.gcn = gcn
    self.zca = zca

    if self.directory is None: raise ValueError("Directory not set")
    if self.train_files is None: raise ValueError("Training data files not set")
    if self.test_files is None: raise ValueError("Test data files not set")
    if self.keys is None: raise ValueError("Data and labels keys not set")
    if self.dims is None: raise ValueError("Data dimensions not set")

    train_images, train_labels = [], []
    self.train_num_examples = 0
    for data_file in self.train_files:
      with open(self.directory + data_file, 'rb') as open_file:
        data_dict = pickle.load(open_file, encoding = 'bytes')
      train_images.append(data_dict[self.keys[0]])
      train_labels.append(np.array(data_dict[self.keys[1]]))
      self.train_num_examples += len(data_dict[self.keys[1]])

    test_images, test_labels = [], []
    self.test_num_examples = 0
    for data_file in self.test_files:
      with open(self.directory + data_file, 'rb') as open_file:
        data_dict = pickle.load(open_file, encoding = 'bytes')
      test_images.append(data_dict[self.keys[0]])
      test_labels.append(np.array(data_dict[self.keys[1]]))
      self.test_num_examples += len(data_dict[self.keys[1]])

    train_dims = np.hstack((self.train_num_examples, self.dims))
    self.train_images = np.vstack(train_images).reshape(train_dims)
    self.train_labels = np.hstack(train_labels)
    test_dims = np.hstack((self.test_num_examples, self.dims))
    self.test_images = np.vstack(test_images).reshape(test_dims)
    self.test_labels = np.hstack(test_labels)

    depth_axis = 1 if gcn_within_depth else None
    if self.gcn: # default axes should be fine
      self.train_images = global_contrast_norm(self.train_images, depth_axis = depth_axis)
      self.test_images = global_contrast_norm(self.test_images, depth_axis = depth_axis)

    if self.zca: # default axes should be fine
      self.train_images = zca_whitening(self.train_images)
      self.test_images = zca_whitening(self.test_images)

    self.batch_index = self.train_num_examples # forces a repermutation on first update

    if self.depth_to_last_dim:
      train_dims = np.hstack((train_dims[0], train_dims[2:], train_dims[1]))
      self.train_images = np.swapaxes(np.swapaxes(self.train_images, 1, 3), 1, 2).reshape(train_dims)
      test_dims = np.hstack((test_dims[0], test_dims[2:], test_dims[1]))
      self.test_images = np.swapaxes(np.swapaxes(self.test_images, 1, 3), 1, 2).reshape(test_dims)

#-------------------------------------------------------------------------------
  def train_next_batch(self, batch_size = 1, rand_horz_flip = False, rand_bord_crop = False, verbose = False):
    if self.batch_index is None:
      raise ValueError("Cannot return batches without first calling imager.read_data()")

    if self.batch_index + batch_size >= self.train_num_examples:
      self.train_permute = np.random.permutation(self.train_num_examples)
      self.batch_index = 0

    i, j = self.batch_index, self.batch_index + batch_size
    batch = self.train_permute[i:j]
    self.batch_index = j
    batch_images = self.train_images[batch]
    batch_labels = self.train_labels[batch]

    hflip = None if not rand_horz_flip else np.random.binomial(1, 0.5, batch_size)
    if hflip is not None:
      flip_ax = 1 if self.depth_to_last_dim else 2
      for i in range(batch_size):
        if hflip[i]:
          batch_images[i] = np.flip(batch_images[i], flip_ax)

    bcrop = None if not rand_bord_crop else np.random.randint(-1, 2, size = [batch_size, 2])
    abval = np.array(self.border_val, dtype = batch_images.dtype)
    aug_dims = np.copy(self.dims) 
    aug_dims[1:] += 2
    if self.depth_to_last_dim: aug_dims = np.hstack((aug_dims[1:], aug_dims[0]))
    if bcrop is not None:
      xx_crop_dict = {-1:[0, self.dims[1]], 0:[1, self.dims[1]+1], 1:[2, self.dims[1]+2]}
      yy_crop_dict = {-1:[0, self.dims[2]], 0:[1, self.dims[2]+1], 1:[2, self.dims[2]+2]}
      base_image = np.tile(abval, aug_dims)
      if self.depth_to_last_dim:
        for i in range(batch_size):
          aug_image = np.copy(base_image)
          aug_image[1:-1, 1:-1, :] = batch_images[i]
          xx, yy = xx_crop_dict[bcrop[i, 0]], yy_crop_dict[bcrop[i, 1]]
          batch_images[i] = aug_image[xx[0]:xx[1], yy[0]:yy[1], :]
      else:
        for i in range(batch_size):
          aug_image = np.copy(base_image)
          aug_image[:, 1:-1, 1:-1] = batch_images[i]
          xx, yy = xx_crop_dict[bcrop[i, 0]], yy_crop_dict[bcrop[i, 1]]
          batch_images[i] = aug_image[:, xx[0]:xx[1], yy[0]:yy[1]]

    if verbose:
      print(batch)
      if hflip is not None: print(hflip)
      if bcrop is not None: print(bcrop)
    
    return batch_images, batch_labels

#-------------------------------------------------------------------------------
