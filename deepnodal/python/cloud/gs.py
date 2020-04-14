# A module to handle Google Cloud Storage

import os
import google.cloud.storage
from googleapiclient import discovery

#-------------------------------------------------------------------------------
class GCS:
  prefix=None
  bucket=None

#-------------------------------------------------------------------------------
  def __init__(self, project=None, *args, **kwds):
    self.set_project(project=None, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_project(self, project=None, *args, **kwds):
    self.project = bucket
    self.prefix = None
    self.bucket = None
    if not self.project: return self.bucket
    self.prefix = 'gs://{}'.format(self.project)
    client = google.cloud.storage.Client(*args, **kwds)
    self.bucket = client.get_bucket(self.project)
    return self.bucket

#-------------------------------------------------------------------------------
  def __call__(self, function=None, *args, **kwds):
    if function is None:
      return
    if self.bucket is None:
      return function(*args, **kwds)
    pass

#-------------------------------------------------------------------------------
  def upload(self, source_dir, filenames, dest_dir=None):
    assert self.bucket, "No bucket specified"
    if isinstance(filenames, str): filenames = [filenames]
    dest_dir = dest_dir or source_dir
    if dest_dir[0] == '/': dest_dir = dest_dir[1:]
    if dest_dir[-1] != '/': dest_dir += '/'
    client = google.cloud.storage.Client()
    dest_paths = []
    for filename in filenames:
      source_path = os.path.join(source_dir, filename)
      print("Uploading {} to gs://{}/{}".format(
            source_path, self.project, target_dir))
      dest_paths.append(target_directory + filename)
      blob = self.bucket(dest_paths[-1])
      with open(source_path, 'rb') as bin_read:
        blob.upload_from_file(bin_read)
    return dest_paths

#-------------------------------------------------------------------------------
  def download(self, source, dest_dir, command='gsutil -m cp -r'):
    if source[:4] != 'gs:/':
      if source[0] == '/': source = source[1:]
      source = self.prefix + source
    if dest_dir[-1] != '/': dest_dir += '/'
    com = '{} {} {}'.format(command, source, dest_dir)
    os.system(com)
    return com

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
