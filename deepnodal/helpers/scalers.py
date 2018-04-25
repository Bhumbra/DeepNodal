# A Python module for centering/scaling data contained in Numpy arrays.
#
# Gary Bhumbra.

#--------------------------------------------------------------------------------
import numpy as np
import pickle

#--------------------------------------------------------------------------------
FLOAT_EPSILON = 1e-11

#--------------------------------------------------------------------------------
def middle_scale(_data, locat_ax = None, scale_ax = None, locat_val = 0., scale_val = 2., dtype = float):
  # Linearly transforms data to be centered and scaled to have a minimum of -1 and maximum of +1)

  data = _data.astype(dtype)
  dims = np.atleast_1d(data.shape)
  all_axes = np.arange(dims.ndim, dtype = int)

  locat_ax = all_axes if locat_ax is None else np.atleast_1d(locat_ax)
  scale_ax = locat_ax if scale_ax is None else np.atleast_1d(scale_ax)

  # Middle
  if len(locat_ax):
    locat_dims = np.copy(dims)
    locat_dims[locat_ax] = 1
    maxim_data = np.max(data, axis = tuple(locat_ax))
    minim_data = np.min(data, axis = tuple(locat_ax))
    locat_data = 0.5 * (maxim_data + minim_data).reshape(locat_dims)
    data -= locat_data

  # Scale
  if len(scale_ax):
    scale_dims = np.copy(dims)
    scale_dims[locat_ax] = 1
    maxim_data = np.max(data, axis = tuple(scale_ax))
    minim_data = np.min(data, axis = tuple(scale_ax))
    scale_data = 0.5 * (maxim_data - minim_data).reshape(scale_dims)
    data /= (scale_data + FLOAT_EPSILON)
  
  if len(scale_ax) and scale_val != 2.:
    data *= 0.5 * scale_val

  if len(locat_ax) and locat_val != 0.:
    data += locat_val

  return data

#--------------------------------------------------------------------------------
def center_scale(_data, locat_ax = None, scale_ax = None, locat_val = 0., scale_val = 1., dtype = float):
  # Linearly transforms data to have a mean of 0. and standard deviation (and thus variance) of 1.

  data = _data.astype(dtype)
  dims = np.atleast_1d(data.shape)
  all_axes = np.arange(dims.ndim, dtype = int)

  locat_ax = all_axes if locat_ax is None else np.atleast_1d(locat_ax)
  scale_ax = locat_ax if scale_ax is None else np.atleast_1d(scale_ax)

  # Center
  if len(locat_ax):
    locat_dims = np.copy(dims)
    locat_dims[locat_ax] = 1
    locat_data = np.mean(data, axis = tuple(locat_ax)).reshape(locat_dims)
    data -= locat_data

  # Scale
  if len(scale_ax):
    scale_dims = np.copy(dims)
    scale_dims[scale_ax] = 1
    scale_data = np.std(data, axis = tuple(scale_ax)).reshape(scale_dims)
    data /= (scale_data + FLOAT_EPSILON)
  
  if len(scale_ax) and scale_val != 1.:
    data *= scale_val

  if len(locat_ax) and locat_val != 0.:
    data += locat_val

  return data

#--------------------------------------------------------------------------------

