# Module to monitor GPU temps

#-------------------------------------------------------------------------------
import subprocess
import time
DEF_COM = 'nvidia-smi'
DEF_COM_ARGS = ['--query-gpu=index,temperature.gpu', '--format=csv,noheader,nounits']
DEF_MIN_POLL = 1.

#-------------------------------------------------------------------------------
class GPUTemps:
  _com = None
  _com_args = None
  _call = None
  _stdout = None
  _stderr = None
  _min_poll = None
  _max_wait = None
  __t0 = None

#-------------------------------------------------------------------------------
  def __init__(self, com=None, *com_args):
    self.set_com(com, *com_args)
    self.set_polling()

#-------------------------------------------------------------------------------
  def set_com(self, com=None, *args):
    self._com = com
    self._com_args = tuple(args)
    if self._com is None:
      assert not self._com_args,\
        "Cannot assign command args without command"
      self._com = DEF_COM
      self._com_args = tuple(DEF_COM_ARGS)
    self._command = [self._com] + list(self._com_args)

#-------------------------------------------------------------------------------
  def set_polling(self, min_poll=DEF_MIN_POLL, max_wait=None):
    self._min_poll = min_poll
    self._max_wait = max_wait
    self.__t0 = time.time() - 2.*self._min_poll

#-------------------------------------------------------------------------------
  def __call__(self):
    t = time.time()
    if t - self.__t0 < self._min_poll:
      return self._stdout
    self.__t0 = t

    try:
      self._call = subprocess.run(self._command, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
      self._stdout = self._call.stdout
      self._stderr = self._call.stderr
    except BaseException:
      self._call = None
      self._stdout = None
      self._stderr = None

    if self._stdout:
      self._stdout = [string for string in self._stdout.decode('utf-8').split(\
                                           '\n') if string]
    return self._stdout

#-------------------------------------------------------------------------------
  def ret_temps(self):
    stdout = self.__call__() or self._stdout
    temps = {}
    if stdout:
      for string in stdout:
        keyval = string.replace(' ', '').split(',')
        temps.update({int(keyval[0]): int(keyval[1])})
    return temps

#-------------------------------------------------------------------------------
  def wait_not_above(self, max_temp=None):
    if not max_temp: return None
    start = time.time()
    wait = start - self.__t0
    done = wait < self._min_poll
    while not done:
      temps = list(self.ret_temps().values())
      if not temps:
        return wait
      done = max(temps) <= max_temp
      wait = time.time() - start
      if not done:
        if self._max_wait:
          if wait > self._max_wait:
            raise ValueError("Maximum wait {} exceeded".format(self._max_wait))
        time.sleep(self._min_poll)
    return wait
  
#-------------------------------------------------------------------------------
