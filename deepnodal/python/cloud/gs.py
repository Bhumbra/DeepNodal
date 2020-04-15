# A module to handle Google Cloud Storage

import os
import subprocess
import csv
import google.cloud.storage
import deepnodal
from googleapiclient import discovery

#-------------------------------------------------------------------------------
TMP_DIR = '/tmp'
CLEAR_TMP = True
GCML_TRAINING_INPUT_KEYWORDS = {
                                'masterType',
                                'packageURIs', 
                                'pythonModule',
                                'region',
                                'args',
                                'jobDir',
                                'runtimeVersion',
                                'pythonVersion',
                                'jobId',
                                'trainingInput'
                               }

#-------------------------------------------------------------------------------
class GCS:
  prefix=None
  bucket=None

#-------------------------------------------------------------------------------
  def __init__(self, project=None, *args, **kwds):
    self.set_project(project, *args, **kwds)

#-------------------------------------------------------------------------------
  def _maybe_unprefix(self, path):
    if path[:5] == 'gs://':
      path = path[5:]
    else:
      if path[0] != '/':
        path = '/' + path
      return path
    split_path = path.split('/')
    composite_path = '/'.join(split_path[1:])
    if path[-1] == '/': return composite_path + '/'
    return composite_path

#-------------------------------------------------------------------------------
  def set_project(self, project=None, *args, **kwds):
    self.project = project
    self.prefix = None
    self.bucket = None
    if not self.project: return self.bucket
    if self.project[:5] == 'gs://':
      self.project = self.project[5:].split('/')[0]
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
    
    path, spec = self._maybe_unprefix(args[0]), args[1]
    directory, filename = os.path.split(path)
    tmp_dir = self.ret_tmp_dir(directory + '/')
    tmp_path = os.path.join(tmp_dir, filename)
    if not os.path.isdir(tmp_dir):
      os.mkdir(tmp_dir)

    # Download then read
    if spec in ['rb', 'read']:
      self.download(path, tmp_dir)
      if spec == 'rb':
        with open(tmp_path, spec) as read_bin:
          out = function(read_bin, **kwds)
      else:
        out = function(tmp_path,  **kwds)

    # Write then upload
    elif spec in ['wb', 'write']:
      if spec == 'wb':
        with open(tmp_path, spec) as write_bin:
          out = function(write_bin, **kwds)
      else:
        out = function(tmp_path, **kwds)
      self.upload(tmp_dir, filename, directory)

    # Otherwise unknown read/write spec
    else:
      raise ValueError("Unknown args[1] spec: {}".format(spec))

    if CLEAR_TMP: os.remove(tmp_path)
    return out

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
    source = self._maybe_unprefix(source)
    source = self.prefix + source
    if dest_dir[-1] != '/': dest_dir += '/'
    com = '{} {} {}'.format(command, source, dest_dir)
    os.system(com)
    return com

#-------------------------------------------------------------------------------
  def package(self, modules, dest_dir):
    def _setup(directory, 
               python_path='/usr/bin/python3',
               setup_py='setup.py',
               sdist='sdist'):
      cwd = os.getcwd()
      os.chdir(directory)
      subprocess.call([python_path, setup_py, sdist])
      dist_dir = directory if not sdist else os.path.join(directory, 'dist')
      list_dir = os.listdir(dist_dir)
      os.chdir(cwd)
      return os.path.join(dist_dir, list_dir[0])

    assert self.prefix, "Project path needed"
    if not isinstance(modules, dict):
      mod_path = lambda module: '/'.join(module.__path__[0].split('/')[:-1]) 
      if isinstance(modules, (list, tuple)):
        modules = {module: mod_path(module) for module in modules}
      else:
        modules = {modules: mod_path(modules)}
    uris = {}
    for module, path in modules.items():
      dist = os.path.join(path, _setup(path))
      dist_dir, dist_file = os.path.split(dist)
      uris.update({module: os.path.join(self.prefix, dest_dir, dist_file)})
      self.upload(dist_dir, dist_file, dest_dir)
    return uris

#-------------------------------------------------------------------------------
  def package_dn(self, dest_dir='deepnodal'):
    return self.package(deepnodal, dest_dir)

#-------------------------------------------------------------------------------
  def cloud_ml(self, proj_dir, *args, **kwds):
    args = tuple(args) or 'ml', 'v1'
    kwds = dict(kwds)
    assert 'body' not in kwds, "cloudml requires body={} keyword"
    body = kwds['body']
    assert 'jobId' in body, "Input body dictionary must contain jobId"
    assert 'trainingInput' in body, "Input body dictionary must contain trainingInput"
    missing = []
    for kwd in GCML_TRAINING_INPUT_KEYWORDS:
      if kwd not in body['trainingInput']:
        missing.append(kwd)
    assert not missing, "Following trainingInput keys missing: {}".\
        format(','.join(missing))
    return discovery(*args).projects().jobs().create(**kwds)

#-------------------------------------------------------------------------------
