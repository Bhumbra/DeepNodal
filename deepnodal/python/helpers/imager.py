# A generic image handler for machine learning

# Gary Bhumbra.

#--------------------------------------------------------------------------------
import numpy as np
import pickle
from deepnodal.python.helpers.scalers import *
from deepnodal.python.helpers.batcher import *

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
class imager (batcher):
  keys = None
  dims = None
  depth_to_last_dim = None
  border_val = None
  gcn = None
  zca = None

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], set_spec=[], files=[], directory='',
                     dims=None, depth_to_last_dim=False, border_val=0):
    super().__init__(set_names, set_spec, files, directory)
    self.set_dims(dims, depth_to_last_dim, border_val)

#-------------------------------------------------------------------------------
  def set_dims(self, dims = None, depth_to_last_dim = False, border_val = 0):
    self.dims = dims
    self.depth_to_last_dim = depth_to_last_dim
    self.border_val = border_val

#-------------------------------------------------------------------------------
  def read_data(self, input_spec, label_spec, 
                      gcn=False, zca=False, gcn_within_depth=True):
    if self.dims is None: raise ValueError("Data dimensions not set")
    if type(input_spec) is str and type(label_spec) is str:
      inputs, labels = super().read_data(input_spec, label_spec)
    else:
      inputs, labels = input_spec, label_spec
    inputs = self._preprocess(inputs)
    return self.set_data(inputs, labels)

#-------------------------------------------------------------------------------
  def _preprocess(self, inputs, gcn=False, zca=False, gcn_within_depth=True):
    self.gcn = gcn
    self.zca = zca
    self.gcn_within_depth = gcn_within_depth
    dims = np.hstack([len(inputs), self.dims])
    inputs = np.reshape(inputs, dims)
    depth_axis = 1 if self.gcn_within_depth else None
    if self.gcn:
      inputs = global_contrast_norm(inputs, depth_axis=depth_axis)
    if self.zca:
      inputs = zca_whitening(inputs)
    if self.depth_to_last_dim:
      dims = np.hstack((dims[0], dims[2:], dims[1]))
      inputs = np.swapaxes(np.swapaxes(inputs, 1, 3), 1, 2).reshape(dims)
    return inputs

#-------------------------------------------------------------------------------
  def _postprocess(self, inputs=None, rand_horz_flip=False, rand_bord_crop=False):
    if inputs is None: return inputs
    inputs = np.array(inputs)
    batch_size = len(inputs)

    # Horizonal flip
    hflip = None if not rand_horz_flip else np.random.binomial(1, 0.5, batch_size)
    if hflip is not None:
      flip_ax = 1 if self.depth_to_last_dim else 2
      for i in range(batch_size):
        if hflip[i]:
          inputs[i] = np.flip(inputs[i], flip_ax)

    # Border crop
    bcrop = None if not rand_bord_crop else np.random.randint(-1, 2, size = [batch_size, 2])
    abval = np.array(self.border_val, dtype = inputs.dtype)
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
          aug_image[1:-1, 1:-1, :] = inputs[i]
          xx, yy = xx_crop_dict[bcrop[i, 0]], yy_crop_dict[bcrop[i, 1]]
          inputs[i] = aug_image[xx[0]:xx[1], yy[0]:yy[1], :]
      else:
        for i in range(batch_size):
          aug_image = np.copy(base_image)
          aug_image[:, 1:-1, 1:-1] = batch_images[i]
          xx, yy = xx_crop_dict[bcrop[i, 0]], yy_crop_dict[bcrop[i, 1]]
          inputs[i] = aug_image[:, xx[0]:xx[1], yy[0]:yy[1]]

    return inputs

#-------------------------------------------------------------------------------
  def next_batch(self, set_name=DEFAULT_SET_NAME, batch_size=None, randomise=None,
                 rand_horz_flip=False, rand_bord_crop=False):
    data = super().next_batch(set_name, batch_size, randomise)
    if data is None:
      return data
    inputs, labels = data
    inputs = self._postprocess(inputs, rand_horz_flip, rand_bord_crop)
    return inputs, labels

#-------------------------------------------------------------------------------
