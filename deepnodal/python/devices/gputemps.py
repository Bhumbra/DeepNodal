# Module to monitor GPU temps

#-------------------------------------------------------------------------------
import subprocess
DEF_COM = 'nvidia-smi'
DEF_COM_ARGS = ['--query-gpu=index,temperature.gpu', '--format=csv,noheader,nounits']

#-------------------------------------------------------------------------------
class GPUTemps:
  _com = None
  _com_args = None
  _call = None
  _stdout = None
  _stderr = None

#-------------------------------------------------------------------------------
  def __init__(self, com=None, *com_args):
    self.set_com(com, *com_args)
    self.set_poling()

#-------------------------------------------------------------------------------
  def set_com(self, com=None, *args):
    self._com = com
    self._com_args = tuple(args)
    if self._com is None:
      assert not self._com_args,\
        "Cannot assign command args without command"
      self._com = DEF_COM
      self._com_args = tuple(DEF_COM_ARGS)

#-------------------------------------------------------------------------------
  def set_poling(self, poling=1.):
    self._poling = poling

#-------------------------------------------------------------------------------
  def __call__(self):
    command = [self._com] + list(self._com_args)
    self._call = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    self._stdout = self._call.stdout
    self._stderr = self._call.stderr
    if self._stdout:
      self._stdout = [string for string in self._stdout.decode('utf-8').split(\
                                           '\n') if string]
    return self._stdout

#-------------------------------------------------------------------------------
  def ret_temps(self):
    stdout = self.__call__()
    temps = {}
    for string in stdout:
      keyval = string.replace(' ', '').split(',')
      temps.update({int(keyval[0]): int(keyval[1])})
    return temps

#-------------------------------------------------------------------------------
