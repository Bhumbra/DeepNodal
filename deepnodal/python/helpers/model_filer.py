"""
A class to look after saving and loading BRU models.
"""

import os 
import fnmatch

SLASH = '/'

def lsdir(dn = None, spec = [], opts = 0): # opts = 0 files, 1 = files+stems, 2 = files+stems+extns
  if dn is None: dn = os.getcwd()
  if type(spec) is str:
    if len(spec): spec = [spec]
  fn = os.listdir(dn)
  n = len(fn)
  N = len(spec)
  if not(N):
    if not(opts): return fn
    stem = [''] * n
    extn = [''] * n
    for i in range(n):
      stem[i], extn[i] = os.path.splitext(fn[i])
    if opts == 1: return fn, stem
    return fn, stem, extn
  filn = []
  stem = []
  extn = []
  for i in range(n):
    ste, ext = os.path.splitext(fn[i])
    addfn= False
    for j in range(N):
      sp = spec[j]
      isextn = False
      if len(sp):
        if sp[0] == '.': isextn = True
      if not(addfn):
        addfn = ext.lower() == sp.lower() if isextn else (len(fnmatch.filter([fn[i]], sp)) > 0)
    if addfn:
      filn.append(fn[i])
      stem.append(ste)
      extn.append(ext)
  if not(opts): return filn
  if opts == 1: return filn, stem
  return filn, stem, extn

class model_filer (object):
  outer = None        # outer directory
  model = None        # model name
  altdn = None        # alternative directories

  def __init__(self, outer = None, model = None):
    self.outer = outer
    self.model = model
    if self.outer is None: return
    if self.outer[-1] != SLASH: self.outer +=SLASH
    if self.model is None: return
    self.check_models()

  def check_models(self):
    self.altdn = lsdir(self.outer, self.model + '*')
    return self.ret_altdn()

  def ret_altdn(self):
    return self.altdn
 
  def interview(self):
    if self.altdn is None: return
    if not(len(self.altdn)): return
    for i, altdir in enumerate(self.altdn):
      print(str(i) + ": " + altdir)
    inp = input("Select # directory (or return to start anew): ")
    moddn = None if not len(inp) else self.altdn[int(inp)]

    if moddn is None: return

    ckpts = lsdir(self.outer + moddn + "/", "*.meta")
    if not(len(ckpts)):
      print("No checkpoint found - starting anew")
      return
    ckpts = [ckpt.replace(".meta", "") for ckpt in ckpts]
    for i, ckpt in enumerate(ckpts):
      print(str(i) + ": " + ckpt)
    inp = input("Select # model (or return for final listed): ")
    inp = len(ckpts)-1 if not(len(inp)) else int(inp)
    return self.outer + moddn + "/" + ckpts[inp]

