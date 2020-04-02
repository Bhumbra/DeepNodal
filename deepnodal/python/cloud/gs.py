# A module to handle Google Cloud Storage

import os
# import google.cloud.storage

#-------------------------------------------------------------------------------
class GCS:
  bucket = None

#-------------------------------------------------------------------------------
  def __init__(self, bucket=None):
    self.bucket = bucket

#-------------------------------------------------------------------------------
  def __call__(self, function=None, *args, **kwds):
    if function is None:
      return
    if self.bucket is None:
      return function(*args, *kwds)
    pass

#-------------------------------------------------------------------------------
  def upload(self, source_dir, filenames, dest_dir=None):
    pass

#-------------------------------------------------------------------------------
  def read(self, filename):
    pass

#-------------------------------------------------------------------------------
  def package(self, modules, dest_dir):
    pass

#-------------------------------------------------------------------------------
  def cloudml(self, proj_dir, *args, **kwds):
    pass

#-------------------------------------------------------------------------------
