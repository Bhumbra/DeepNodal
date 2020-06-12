# A checkpoint manager for TensorBoard logs

#-------------------------------------------------------------------------------
import os
import glob
import collections
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#-------------------------------------------------------------------------------
class Checkpointer:

#-------------------------------------------------------------------------------
  # Public
  evts_stem = 'events.out.tfevents.' # Events stem
  meta_extn = 'meta'
  directory = None
  subprefix = None                   # Optional filter for subdirectories

  # Protected
  _subdirs = None

#-------------------------------------------------------------------------------
  def __init__(self, evts_stem=None, meta_extn=None):
    self.evts_stem = evts_stem or self.evts_stem
    self.meta_extn = meta_extn or self.meta_extn

#-------------------------------------------------------------------------------
  def set_directory(self, directory=None, subprefix=None):
    self.directory = directory
    return self.ret_subdirs(subprefix)

#-------------------------------------------------------------------------------
  def ret_subdirs(self, subprefix=None):
    self.subprefix = subprefix
    if not self.directory:
      return self._subdirs
    if isinstance(self.subprefix, (list, tuple)):
      self._subdirs  = list(self.subprefix)
      return self._subdirs
    self._subdirs = None
    if not self.subprefix:
      self._subdirs = ['']
      return self._subdirs
    glob_path = os.path.join(self.directory, self.subprefix+'*')
    candidates = sorted(glob.glob(glob_path))
    self._subdirs = []
    for candidate in candidates:
      if os.path.isdir(candidate):
        _, subdir = os.path.split(candidate)
        self._subdirs.append(subdir)
    return self._subdirs

#-------------------------------------------------------------------------------
  def ret_checkpoints(self):
    """ Returns checkpoints as an ordered dictionary of ordered dictionaries,
    in the form {subdir: {checkpoint_file_stem: numeric_progress}}
    """
    checkpoints = collections.OrderedDict()
    if not self._subdirs: return checkpoints
    for subdir in self._subdirs:
      checkpoints.update({subdir: collections.OrderedDict()})
      meta_path = os.path.join(self.directory, subdir) + '/*.' + self.meta_extn
      ckpt_paths = glob.glob(meta_path)
      ckpt_progs = [None] * len(ckpt_paths)
      ckpt_stems = [''] * len(ckpt_paths)
      for i, ckpt_path in enumerate(ckpt_paths):
        _, file_name = os.path.split(ckpt_path)
        file_stem, _ = os.path.splitext(file_name)
        progress = file_stem.split('-')[-1]
        assert progress.isnumeric(), "Non-numeric progress suffix: {}".\
                                     format(progress)
        ckpt_stems[i] = file_stem
        ckpt_progs[i] = int(progress)
      ckpt_order = np.argsort(ckpt_progs).tolist()
      for index in ckpt_order:
        checkpoints[subdir].update({ckpt_stems[index]: ckpt_progs[index]})
    return checkpoints

#-------------------------------------------------------------------------------
  def read_scalars(self, prefix, *args, collate=None):
    """ Returns a dictionary containing the scalar values requested within
    the scalar names in args, under prefix (e.g. 'supervisor/metrics').
    Collate can be 'off', 'all', or 'canonical' (default).
    """
    if prefix[-1] != '/': prefix += '/'
    for arg in args:
      assert isinstance(arg, str),\
        "Scalar names must be strings, found: {}".format(arg)
    subdirs = list(self.ret_checkpoints().keys())
    data = collections.OrderedDict()
    for subdir in subdirs:
      evts_path = os.path.join(self.directory, subdir, self.evts_stem+'*')
      evts_file  = glob.glob(evts_path)
      if not evts_file: continue
      assert len(evts_file) == 1, "Multiple events file found: {}".format(evts_file)
      evts_file = evts_file[0]
      print("Reading: {}".format(evts_file))
      EvtAcc = EventAccumulator(evts_file)
      EvtAcc.Reload()
      data.update({subdir: collections.OrderedDict()})
      for arg in args:
        data[subdir].update({arg: collections.OrderedDict()})
        wall_time, global_step, scalars = zip(*EvtAcc.Scalars(prefix + arg))
        scalar_dict = collections.OrderedDict()
        data[subdir][arg].update({'wall_time': wall_time})
        data[subdir][arg].update({'global_step': global_step})
        data[subdir][arg].update({'scalars': scalars})

    if collate == False or collate == 'off':
      return data

    # Collate tables
    tables = collections.OrderedDict()
    if not data:
      return data
    for arg in args:
      tables.update({arg: collections.OrderedDict()})
      for key in ['wall_time', 'global_step', 'scalars']:
        composite = np.concatenate([data[subdir][arg][key] \
                                   for subdir in subdirs])
        if key == 'global_step':
          global_step_0 = np.min(composite)
          if np.sum(composite == global_step_0) > 1:
            print("Warning: {} scalar events indicate multiple starts \
                   from step {} for runs: {}".\
                  format(arg, global_step_0, subdirs))
        tables[arg].update({key: composite})
    if collate == True or collate == 'all':
      return tables

    # Remove back-tracks
    for arg in args:
      global_step = tables[arg]['global_step']
      back_tracks = np.nonzero(np.diff(global_step)<=0)[0]
      if not len(back_tracks):
        continue
      forward = np.ones(len(global_step), dtype=bool)
      back_tracks = back_tracks[::-1] + 1
      for index in back_tracks:
        forward[:index] = global_step[:index] < global_step[index]
      global_step = global_step[forward]
      assert np.min(np.diff(global_step)) > 0,\
          "Failed in removing back-tracks"
      for key in ['wall_time', 'global_step', 'scalars']:
        tables[arg][key] = tables[arg][key][forward]
    return tables

#-------------------------------------------------------------------------------
  def read_distros(self, prefix, *args):
    if prefix[-1] != '/': prefix += '/'
    for arg in args:
      assert isinstance(arg, str),\
        "Scalar names must be strings, found: {}".format(arg)
    subdirs = list(self.ret_checkpoints().keys())
    data = collections.OrderedDict()
    for subdir in subdirs:
      evts_path = os.path.join(self.directory, subdir, self.evts_stem+'*')
      evts_file  = glob.glob(evts_path)
      if not evts_file: continue
      assert len(evts_file) == 1, "Multiple events file found: {}".format(evts_file)
      evts_file = evts_file[0]
      print("Reading: {}".format(evts_file))
      EvtAcc = EventAccumulator(evts_file, size_guidance={'histograms': 0})
      EvtAcc.Reload()
      data.update({subdir: collections.OrderedDict()})
      for arg in args:
        data[subdir].update({arg: collections.OrderedDict()})
        wall_time, global_step, histogram = zip(*EvtAcc.Histograms(prefix + arg))
        distros = [hist[5:] for hist in histogram]
        scalar_dict = collections.OrderedDict()
        data[subdir][arg].update({'wall_time': wall_time})
        data[subdir][arg].update({'global_step': global_step})
        data[subdir][arg].update({'histogram': histogram})
        data[subdir][arg].update({'distros': distros})
    return data

#-------------------------------------------------------------------------------

