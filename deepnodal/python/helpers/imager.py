# A generic image handler for machine learning

# Gary Bhumbra.

#--------------------------------------------------------------------------------
import numpy as np
import pickle
from deepnodal.python.helpers.scalers import *
from deepnodal.python.helpers.batcher import *
#--------------------------------------------------------------------------------
# Map axis according to depth_last_dim
HORZ_FLIP_AXIS = {False: 2, True: 1}
VERT_FLIP_AXIS = {False: 1, True: 0}

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
def random_flip(data, horz_spec=False, vert_spec=False, depth_last_dim=False):
  num_data = len(data)
  horz_axis = None
  vert_axis = None
  horz_flip = None
  vert_flip = None
  if horz_spec:
    horz_axis = HORZ_FLIP_AXIS[self.depth_last_dim]
    horz_flip = np.random.binomial(1, 0.5, num_data)
  if vert_spec:
    vert_axis = VERT_FLIP_AXIS[self.depth_last_dim]
    vert_flip = np.random.binomial(1, 0.5, num_data)
  for i in range(num_data):
    if horz_spec:
      if horz_flip[i]:
        inputs[i] = np.flip(inputs[i], horz_axis)
    if vert_spec:
      if vert_flip[i]:
        inputs[i] = np.flip(inputs[i], vert_axis)
  return inputs

#-------------------------------------------------------------------------------
def random_crop(data, horz_spec=0, vert_spec=0, depth_last_dim=False):
  if not horz_spec and not vert_spec:
    return data
  num_data = data.shape[0]
  horz_crop = None
  vert_crop = None
  if horz_spec:
    horz_crop = np.random.randint(0, 2*horz_spec+1, size=num_data)
  if vert_spec:
    vert_crop = np.random.randint(0, 2*vert_spec+1, size=num_data)
  if not depth_last_dim:
    width, height = data.shape[3], data.shape[2]
    for i in range(num_data):
      datum = data[i]
      if horz_spec:
        crop = horz_crop[i]
        if horz_spec != crop:
          lhs = np.flip(datum[:, :,  :horz_spec], 2)
          rhs = np.flip(datum[:, :, -horz_spec:], 2)
          datum = np.concatenate([lhs, datum, rhs], axis=2)
          datum = datum[:, :, crop:crop+width]
      if vert_spec:
        crop = vert_crop[i]
        if vert_spec != crop:
          top = np.flip(datum[:,  :vert_spec, :], 1)
          bot = np.flip(datum[:, -vert_spec:, :], 1)
          datum = np.concatenate([top, datum, bot], axis=1)
          datum = datum[:, crop:crop+height, :]
      data[i] = datum
  else:
    width, height = data.shape[2], data.shape[1]
    for i in range(num_data):
      datum = data[i]
      if horz_spec:
        crop = horz_crop[i]
        if horz_spec != crop:
          lhs = np.flip(datum[:, :horz_spec,  :], 1)
          rhs = np.flip(datum[:, -horz_spec:, :], 1)
          datum = np.concatenate([lhs, datum, rhs], axis=1)
          datum = datum[:, crop:crop+width, :]
      if vert_spec:
        crop = vert_crop[i]
        if vert_spec != crop:
          top = np.flip(datum[ :vert_spec, :, :], 0)
          bot = np.flip(datum[-vert_spec:, :, :], 0)
          datum = np.concatenate([top, datum, bot], axis=0)
          datum = datum[crop:crop+height, :, :]
      data[i] = datum
  return data
 
#-------------------------------------------------------------------------------
class imager (batcher):
  keys = None
  dims = None
  depth_last_dim = None
  gcn = None
  zca = None

#-------------------------------------------------------------------------------
  def __init__(self, set_names=[], set_spec=[], files=[], directory='',
                     dims=None, depth_last_dim=False):
    super().__init__(set_names, set_spec, files, directory)
    self.set_dims(dims, depth_last_dim)

#-------------------------------------------------------------------------------
  def set_dims(self, dims=None, depth_last_dim=False):
    self.dims = dims
    self.depth_last_dim = depth_last_dim

#-------------------------------------------------------------------------------
  def read_data(self, input_spec, label_spec, 
                      gcn=False, zca=False, gcn_within_depth=True):
    if self.dims is None: raise ValueError("Data dimensions not set")
    if isinstance(input_spec, bytes) and isinstance(label_spec, bytes):
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
    if self.depth_last_dim:
      dims = np.hstack((dims[0], dims[2:], dims[1]))
      inputs = np.swapaxes(np.swapaxes(inputs, 1, 3), 1, 2).reshape(dims)
    return inputs

#-------------------------------------------------------------------------------
  def _postprocess(self, inputs=None, rand_flip=False, rand_crop=False):
    """
    Post-processes inputs according with XY transformations randomly applied
    """
    if inputs is None: return inputs
    if type(inputs) is not np.ndarray:
      inputs = np.array(inputs)
    batch_size = len(inputs)

    # Reflection flip
    if rand_flip:
      if type(rand_flip) is bool:
        rand_flip = [rand_flip, False]
      else: 
        assert len(rand_flip) == 2, "Input rand_flip must be of length 2"
      inputs = random_flip(inputs, rand_flip[0], rand_flip[1], self.depth_last_dim)

    # Border Crop
    if rand_crop:
      if type(rand_crop) is bool:
        rand_crop = 1
      if type(rand_crop) is int:
        rand_crop = [rand_crop, rand_crop]
      else:
        assert len(rand_crop) == 2, "Input rand_crop must be of length 2"
      inputs = random_crop(inputs, rand_crop[0], rand_crop[1], self.depth_last_dim)

    return inputs

#-------------------------------------------------------------------------------
  def next_batch(self, set_name=DEFAULT_SET_NAME, batch_size=None, randomise=None,
                 rand_flip=False, rand_crop=False):
    data = super().next_batch(set_name, batch_size, randomise)
    if data is None:
      return data
    inputs, labels = data
    inputs = self._postprocess(inputs, rand_flip, rand_crop)
    return inputs, labels

#-------------------------------------------------------------------------------
