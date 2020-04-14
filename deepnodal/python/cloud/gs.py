# A module to handle Google Cloud Storage

import os
import google.cloud.storage
from googleapiclient import discovery

TMP_DIR = '/tmp'

#-------------------------------------------------------------------------------
class GCS:
  prefix=None
  bucket=None

#-------------------------------------------------------------------------------
  def __init__(self, project=None, *args, **kwds):
    self.set_project(project, *args, **kwds)

#-------------------------------------------------------------------------------
  def set_project(self, project=None, *args, **kwds):
    self.project = project
    self.prefix = None
    self.bucket = None
    if not self.project: return self.bucket
    if self.project[:5] == 'gs://':
      self.project = self.project[:5].split('/')[0]
    self.prefix = 'gs://{}'.format(self.project)
    client = google.cloud.storage.Client(*args, **kwds)
    self.bucket = client.get_bucket(self.project)
    return self.bucket

#-------------------------------------------------------------------------------
  def ret_tmp_dir(self, path):
    assert path[-1] == '/', "Last character must be '/'"
    if path[0] != '/':
      path = '/' + path
    return TMP_DIR + path

#-------------------------------------------------------------------------------
  def __call__(self, function=None, *args, **kwds):
    """
    If args is entered, it must be contain exactly 2 string elements:
    args[0] = absolute path of file to read or write (excluding project)
    args[1] = either 'rb' or 'wb' weather reading or writing
    """
    if function is None:
      return
    assert len(args) == 2, "{} args entered, 2 expected".format(len(args))
    assert isinstance(args[0], str), "args[0] is not a string"
    assert isinstance(args[1], str), "args[1] is not a string"

    if self.bucket is None:
      if args[1] in ['rb', 'wb']:
        with open(*args) as open_bin:
          out = function(open_bin, **kwds)
        return out
      return function(args[0], **kwds)
    
    if path[0] != '/':
      path = '/' + path
    directory, filename = os.path.split(path)
    tmp_dir = self.ret_tmp_dir(directory + '/')
    tmp_path = os.join(tmp_dir, filename)
    if not os.path.isdir(tmp_dir):
      os.mkdir(tmp_dir)

    # Download then read
    if args[1] in ['rb', 'read']:
      self.download(args[0], tmp_dir)
      if args[1] == 'rb':
        with open(tmp_path, args[1]) as read_bin:
          out = function(read_bin, **kwds)
        return out
      return function(tmp_path,  **kwds)

    # Write then upload
    if args[1] in ['wb', 'write']:
      if args[1] == 'wb':
        with open(tmp_path, args[1]) as write_bin:
          out = function(write_bin, **kwds)
      else:
        out = function(tmp_path, **kwds)
      self.upload(tmp_dir, filename, directory)
      return out

    raise ValueError("Unknown args[1] spec: {}".format(args[1]))

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
            source_path, self.project, dest_dir))
      dest_paths.append(dest_dir + filename)
      blob = self.bucket.blob(dest_paths[-1])
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
