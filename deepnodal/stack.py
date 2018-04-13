# Multilevel 'stack' module for Tensorflow.

# Gary Bhumbra

#-------------------------------------------------------------------------------
from level import *

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class stack (level):
  Arch = None
  # A stack may be a single level or multiple levels - however all streams at the
  # first _and_ final level must have single-stream inputs and outputs respectively
  def __init__(self, dev = None, Arch = None, name = 'stack'):
    level.__init__(self, dev, None, name)
    self.setarch(Arch)

#-------------------------------------------------------------------------------
  def setarch(self, Arch = None):
    self.Arch = Arch
    self.nArch = 0
    if self.Arch is None: return
    if type(self.Arch) is not list:
      self.Arch = [self.Arch]
    self.nArch = len(self.Arch)
    self.narch = self.nArch - 1 # exclude inputs
    if self.narch < 0: return
    self.arch = self.Arch[1:]
    self.SL = self.narch == 1
    if self.SL: 
      self.setarc(self.arch[0])

    return inilevels()

#-------------------------------------------------------------------------------
  def inilevels(self):
    self.SL = self.narch == 1
    self.levels = [None] * self.narch
    if self.narch == 1:
      self.levels[0] = self
    else:
      for i in range(self.narch):
        self.levels[i] = level(self.dev, name = self.name + "/level_" + str(i))
    return self.levels

#-------------------------------------------------------------------------------
  def setist(self, *args, **kwargs):
    argout = level.setist(self, *args, **kwargs)
    if self.SL or self.ist is None: return argout
    return [_level.setist(self.ist) for _level in self.levels]

#-------------------------------------------------------------------------------
  def setipc(self, ipc = ''):
    self.ipc = ipc if type(ipc) is not list else [ipc] * self.narch
    return [_level.setipc(_ipc) for _level, _ipc in zip(self.levels, self.ipc)]

#-------------------------------------------------------------------------------
  def setwin(self, win = None, **win_kwargs):
    self.win = win if type(win) is not list else [win] * self.narch
    self.win_kwargs = win_kwargs
    if not(len(self.win_kwargs)): self.win_kwargs = [self.win_kwargs] * self.narch
    return [_level.setwin(_win, *_win_kwargs) for _level, _win, _win_kwargs in\
            zip(self.levels, self.win, self.win_kwargs)]

#-------------------------------------------------------------------------------
  def setpfn(self, pfn = None, **pfn_kwargs):
    self.pfn = pfn if type(pfn) is not list else [pfn] * self.narch
    self.pfn = win_kwargs
    if not(len(self.pfn_kwargs)): self.pfn_kwargs = [self.pfn_kwargs] * self.narch
    return [_level.setpfn(_pfn, *_pfn_kwargs) for _level, _pfn, _pfn_kwargs in\
            zip(self.levels, self.pfn, self.pfn_kwargs)]

#-------------------------------------------------------------------------------
  def settrf(self, trf = None, trf_last = None, trf_pool = True, trf_conv_premaxpool = False):
    if not(trf_pool) and not(trf_conv_premaxpool):
      raise ValueError("Cannot both disable pooling and conv-premaxpool transfer functions")
    if type(trf) is list:
      self.trf = [trf] * self.narch
      self.trf[-1] = trf_last
    else:
      self.trf = trf
      if trf_last is not None:
        self.trf[-1] = trf_last
    if not trf_conv_premaxpool):
      for i in range(self.narch - 1):
        if self.levels[i].atype == "dense" or self.levels[i].atype == "conv":
          if self.levels[i+1].atype == "pool" and self.apf[i+1] == "max":
            self.trf[i] = None
    return [_level.settrf(_trf, trf_pool) for _level, _trf in zip(self.levels, self.trf)]

#-------------------------------------------------------------------------------
  def setwci(self, *args, **kwargs):
    argout = level.setwci(self, *args, **kwargs)
    if self.SL or self.wci is None: return argout
    return [_level.setwci(self.wci) for _level in self.levels]

#-------------------------------------------------------------------------------
  def setbnm(self, bnm = None, **bnm_kwargs):
    self.bnm = bnm if type(bnm) is not list else [bnm] * self.narch
    self.bnm_kwargs = bnm_kwargs
    if not(len(self.bnm_kwargs)): self.bnm_kwargs = [self.bnm_kwargs] * self.narch
    return [_level.setbnm(_bnm, *_bnm_kwargs) for _level, _bnm, _bnm_kwargs in\
            zip(self.levels, self.bnm, self.bnm_kwargs)]

#-------------------------------------------------------------------------------
  def setlrn(self, *args, **kwargs):
    self.lrn = lrn if type(lrn) is not list else [lrn] * self.narch
    self.lrn_kwargs = lrn_kwargs
    if not(len(self.lrn_kwargs)): self.lrn_kwargs = [self.lrn_kwargs] * self.narch
    return [_level.setlrn(_lrn, *_lrn_kwargs) for _level, _lrn, _lrn_kwargs in\
            zip(self.levels, self.lrn, self.lrn_kwargs)]

#-------------------------------------------------------------------------------
  def setdro(self, dro = None, *args, **kwargs):
    if type(dro) is tf.Session and self.dro is not None and len(args):
      dropo = args[0] if type args[0] is list else [args[0]] * self.narch
      return [_level.setdro(dro, _dropo, **kwargs) for _level, _dropo in zip(self.level, dropo)]
    self.dro = dro if type(dro) is not list else [dro] * self.narch
    self.dro_kwargs = kwargs
    if not(len(self.dro_kwargs)): self.dro_kwargs = [self.dro_kwargs] * self.narch
    return [_level.setdro(_dro, *_dro_kwargs) for _level, _dro, _dro_kwargs in\
            zip(self.levels, self.dro, self.dro_kwargs)]

#-------------------------------------------------------------------------------
  def setopc(self, opc = ''):
    self.opc = opc if type(opc) is not list else [opc] * self.narch
    return [_level.setipc(_opc) for _level, _opc in zip(self.levels, self.opc)]
  
#-------------------------------------------------------------------------------
  def setinput(self, func_call = tf.placeholder, *args, **kwargs)
    if self.ipc is None: self.setipc()
    kwds = dict(kwargs)
    if func_call == tf.placeholder and not 'dtype' in kwds:
      kwds.update({'dtype':tf.float32})
    if self.atype[0] == "dense":
      inp_dim = [None, int(np.prod(self.arch[0]))]
    else:
      inp_dim = [None]
      for dim0 in self.arch[0]:
        inp_dim.append(int(dim0))
    if func_call == tf.placeholder and 'dtype' in kwds:
      if self.dev is None:
          self.Input = func_call(kwds['dtype'], name=self.name + "/input", shape = inp_dim)
      else:
        with tf.device(self.dev):
          self.Input = func_call(kwds['dtype'], name=self.name + "/input", shape = inp_dim)
    else:
      if callable(func_call):
        if self.dev is None:
          self.Input = func_call(*args, name=self.name + "/input", **kwargs)
        else:
          with tf.device(self.dev):
            self.Input = func_call(*args, name=self.name + "/input", **kwargs)
      else:
        self.Input = func_call
    if self.levels is not None: self.setup()
    return self.Input
    
#-------------------------------------------------------------------------------
  def setup(self):
    if self.ist is None: self.setist()
    if self.win is None: self.setwin()
    if self.pfn is None: self.setpfn()
    if self.trf is None: self.settrf()
    if self.wci is None: self.setwci()
    if self.bnm is None: self.setbnm()
    if self.lrn is None: self.setlrn()


