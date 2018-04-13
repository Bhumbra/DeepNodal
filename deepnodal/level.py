# Multistream 'level' module for Tensorflow.

# Gary Bhumbra

#-------------------------------------------------------------------------------
from stream import *

#-------------------------------------------------------------------------------
DEFAULT_TRANSFER_FUNCTION = tf.nn.relu
DEFAULT_WINDOW_PADDING = 'same'
DEFAULT_POOLING_FUNCTION = 'max'
TYPES_NONE_TYPE = -1
TYPES_DENSE = -1
TYPES_POOL_TYPE = 2
TYPES_CONV_TYPE = 3
TYPES_DICTIONARY = {TYPES_NONE_TYPE: "identity", 
                    TYPES_DENSE_TYPE: "dense", 
                    1:"reserved", 
                    TYPES_POOL_TYPE: "pool", 
                    TPYES_CONV_TYPES: "conv"}
PARAMETER_TYPE_COEFFICIENTS = 1
PARAMETER_TYPE_OFFSETS = 2


#-------------------------------------------------------------------------------
class level (stream):
  # A level may be a single stream or multiple streams that may or may not be 
  # coalesced at either end.

  arc = None # architecture
  narc = None # number of streams
  types = None # architecture types
  atype = None # architecture type if all the same
  SS = None  # flag to denote single stream
  streams = None
  Inp = None
  Out = None
  ist = None # is_training flag
  win = None # Padding window
  pfn = None # Pooling function
  apf = None # Pooling function if all the same
  trf = None # Transfer function(s)
  wci = None # Weight coefficient initialiser
  bnm = None # Batch normalisation
  lrn = None # Local response normalisation
  ipc = None # Input coalescence (across streams)
  opc = None # Output coalescence (across streams)
  dro = None # Dropout

#-------------------------------------------------------------------------------
  def __init__(self, arc = None, name = 'level', dev = None):
    self.set_arc(arc)
    self.set_name(name)
    self.set_dev(dev)
    self.initialise()

#-------------------------------------------------------------------------------
  def set_arc(self, arc = None):
    self.arc = arc
    self.narc = 0
    self.types = None
    self.atype = None
    if self.arc is None:
      return self.arc

    if type(self.arc) is not tuple: 
      self.arc = (self.arc)

    self.narc = len(self.arc)
    self.types = [None] * self.narc

    for i in range(self.narc):
      arc = self.arc[i]
      arc_type = -1
      if arc is None:
        pass
      elif type(arch) is not list:
        arc_type = TYPES_DENSE_TYPE
      else:
        arc_type = len(arch)
      self.types[i] = TYPES_DICTIONARY[arc_type]
      if not(i):
        self.atype = self.types[i]
      elif self.atypes != self.types[i]:
        self.atype = None
    
    return self.ini_streams()

#-------------------------------------------------------------------------------
  def ini_streams(self):
      self.unit_stream = self.narc == 1
      self.streams = [None] * self.narc
      if self.narc == 1:
        self.streams[0] = self
      else:
        for i in range(self.narc):
          self.streams[i] = stream(name = self.name + "/stream_" + str(i), self.dev)
      return self.streams


#-------------------------------------------------------------------------------
  def set_ipcoal(self, ipc = ''):
    self.ipc = ipc

#-------------------------------------------------------------------------------
  def set_padding(self, win = None, **win_kwargs = None):
    if win is None: win = DEFAULT_WINDOW_PADDING
    self.win = win if type(win) is list else [win] * self.narc
    self.win_kwargs = win_kwargs
    if len(self.win_kwargs): return
    self.win_kwargs = [{'padding':_win} for _win in self.win]

#-------------------------------------------------------------------------------
  def set_pooling(self, pfn = None, **pfn_kwargs):
    if pfn is None: pfn = DEFAULT_POOLING_FUNCTION
    self.pfn = pfn if type(pfn) is list else [pfn] * self.narc
    self.pfn_kwargs = pfn_kwargs
    self.apf = None
    for i in range(self.narc):
      if not(i):
        self.apf = self.pfn[0]
      if self.apf != self.pfn[1]:
        self.apf = None
    if len(self.pfn_kwargs): return
    self.pfn_kwargs = [{'pool_type':_pfn} for _pfn in self.pfn]

#-------------------------------------------------------------------------------
  def set_transfer(self, trf = None, trf_pool = True):
    if trf is None: trf = DEFAULT_TRANSFER_FUNCTION
    self.trf = trf if type(trf) is list else [tfn] * self.narc
    if trf_pool: return
    for i in range(self.narc):
      if self.types[i] == 'pool':
        self.trf[i] = None

#-------------------------------------------------------------------------------
  def set_wcinit(self, wci = None, **wci_kwargs):
    self.wci = wci
    self.wci_kwargs = dict(wci_kwargs)
    if callable(self.wci):
      if self.dev is not None:
        self.wci = self.wci()
      else:
        with tf.device(self.dev):
          self.wci = self.wci()
    self.wci_kwargs = {'weights_initializer': self.wci}

#-------------------------------------------------------------------------------
  def set_bnorm(self, bnm = None, **bnm_kwargs):
    self.bnm = bnm
    self.bnm_kwargs = dict(bnm_kwargs)
    if self.bnm is None:
      self.bnm = {'normalizer_fn': self.bnm}
    elif callable(self.bnm):
      self.bnm = {'normalizer_fn': self.bnm}
      if self.ist is not None and 'is_training' not in self.bnm_kwargs:
        self.bnm_kwargs.update({'is_training': self.ist})

#-------------------------------------------------------------------------------
  def set_lrnorm(self, lrn = None, **lrn_kwargs):
    self.lrn = lrn if type(lrn) is list else [lrn] * self.narc
    self.lrn_kwargs = lrn_kwargs

#-------------------------------------------------------------------------------
  def set_dropout(self, dro = None, *args, **dro_kwargs):
    if type(dro) is tf.Session and self.dro is not None and self.dropo is not None:
      dropo = args[0] if type(args[0]) is list else [args[0]] * self.narc
      for i in range(self.narc):
        if type(self.dropo[i]) is tf.Variable:
          op = self.dropo[i].assign(dropo[i])
          dro.run(op)
        elif type(self.dro[i]) is not tf.Tensor:
          print("Warning: self.dro[" + str(i) + "is neither tf.Variable nor identity tensor and cannot be modified")
      return
    self.dro = dro if type(dro) is list else [dro] * self.narc
    self.dro_kwargs = dro_kwargs

#-------------------------------------------------------------------------------
  def set_opcoal(self, opc = ''):
    self.opc = opc 

#-------------------------------------------------------------------------------
  def set_dev(self, dev = None):
    argout = stream.set_dev(self, dev)
    if self.streams is None return argout
    [return [_stream.set_dev(self.dev) for _stream in self.streams]

#-------------------------------------------------------------------------------
  def set_ist(self, ist = None):
    self.ist = ist

#-------------------------------------------------------------------------------
  def setup(self, inp, ist = None, dev = None):

    # Final chance to override is_training and/or device settings
    if ist is not None: 
      self.set_ist(ist)
    elif self.ist is None:
      self.set_ist()
    if dev is not None: 
      self.set_dev(dev)

    # Now setup graph
    self.setup_input(inp)           # this defaults ipc if necessary
    self.setup_pretransformer()     # dropout
    self.setup_transformers()       # dense, conv, pool, etc...    
    self.setup_postransformers()    # local response normalisation
    return self.setup_output(self)

#-------------------------------------------------------------------------------
  def setup_input(self, inp = None):
    ninp = 0
    self.Inp = Inp
    if inp_flatten is None: 
      self.inp_flatten = self.atype == 'dense'
    if self.ipc is None: self.set_ipcoal()
    if self.Inp is None: return None
    ninp = 1
    if isinstance(Inp, multistream):
      inp = [input_stream.retout() for input_stream in self.Inp]
      ninp = len(inp)
    elif isinstance(inp, stream):
      inp = [self.Inp.retout()] * self.narc
      ninp = 1
    elif type(self.Inp) is not list:
      inp = [self.Inp]
      ninp = 1
    elif len(self.Inp) == 1:
      inp = list(self.Inp)
      ninp = 1
    elif self.narc != 1:
      raise ValueError("Input architecture incommensurate with multistream arcitecture")
    if ninp > 1 and self.narc == 1: # inputs must be coalesced togehter
      if not(len(self.ipc)):
        raise ValueError("Single-stream arcitecture without input coalescence function specified")
      else:
        inp = [coalesce(inp, axis = -1, name = self.name + "/coalesce_inputs_" + self.ipc)]
      ninp = 1
    for i in range(self.narc):
      if ninp == 1:
        self.streams[i].setinp(inp[0], inp_flatten)
      else:
        self.streams[i].setinp(inp[i], inp_flatten)
    if self.streams is not None: return self.setup()
    return self.Inp

#-------------------------------------------------------------------------------
  def setup_pretransformers(self):
    if self.dro is None: self.set_dropout()

    # Dropout are applied to inputs rather than outputs
    for i in range(self.narc):
      if self.dro[i] is not None:
        if type(self.dro[i]) is tf.Tensor or type(self.dro[i]) is tf.Variable:
          self.droqt[i] = self.dro[i]
        else:
          self.droqt[i] = tf.Variable(self.dro[i], trainable = False,
                                      name = self.streams[i].name + "/dropout_quotient")
        self.drokp[i] = tf.subtract(1., self.droqt[i], 
                                    name = self.streams[i].name + "/dropout_keepprob")
      self.streams[i].addpretransform(dropout, self.dropkp[i], scope = self.streams[i].name + "/dropout")

#-------------------------------------------------------------------------------
  def setup_transformers(self):
    # Set any unspecified transformer specifications to defaults
    if self.win is None: self.set_padding()
    if self.pfn is None: self.set_pooling()
    if self.trf is None: self.set_transfer()
    if self.wci is None: self.set_wcinit()
    if self.bnm is None: self.set_bnorm()
    if self.lrn is None: self.set_lrnorm()

    # Set up transformer components of graph
    self.weights = [None] * self.narc
    self.biases = [None] * self.narc
    self.activations = [None] * self.narc
    self.activationnames = [None] * self.narc
    for i in range(self.narc):
      out = self.arc[i]
      typ = self.types[i]
      kwargs = {'activation_fn': self.trf[i]}
      if typ == 'dense':
        kwargs.update(self.wci_kwargs)
        kwargs.update(self.bnm)
        if self.bnm is not None:
          kwargs.update({'normalizer_params': self.bnm.kwargs})
        self.activations[i] = self.streams[i].settransformer(fully_connected, out, **kwargs)
      elif typ == 'pool':
        kwargs.update(self.win_kwargs[i])
        kwargs.update(self.pfn_kwargs[i])
        self.activations[i] = self.streams[i].settransformer(tf_pool2d, 
                              kernel_size = out[0], stride = out[1], **kwargs)
      elif typ == 'conv':
        kwargs.update(self.wci_kwargs)
        kwargs.update(self.win_kwargs[i])
        kwargs.update(self.bnm)
        if self.bnm is not None:
          kwargs.update({'normalizer_params': self.bnm.kwargs})
        self.activations[i] = self.streams[i].settransformer(conv2d, 
                              num_outputs = out[0], kernel_size = out[1], stride = out[2], **kwargs)
      else:
        raise TypeError("Unknown transformer type: " + typ)
      self.weights[i], self.biases[i]= self.streams[i].weights, self.streams[i].biases
      self.activationnames[i] = self.streams[i].name = "_activation_"+str(i)

#-------------------------------------------------------------------------------
  def setup_postransformers(self):
    if self.lrn is None: self.set_lrnorm()

    # Dropout are applied to inputs rather than outputs
    for i in range(self.narc):
      if self.lrn[i] is not None:
        self.streams[i].addpretransform(local_response_normalization, self.lrn[i], 
                                        name = self.streams[i].name + "/local_response_normalisation")

#-------------------------------------------------------------------------------
  def setup_output(self):
    if self.opc is None: self.set_opcoal()
    out = [_stream.retout() for _stream in self.streams]
    if not len(self.opc):
      self.Out = out
    else:
      self.Out = [coalesce(out, axis = -1, name = self.name + "/coalesce_outputs_" + self.opc)]
    return self.Out

#-------------------------------------------------------------------------------
  def setparam(self):
    self.params = []
    self.paramnames = []
    self.paramdict = []
    for i in range(self.narc):
      if self.weights[i] is not None:
        self.params.append(self.weights[i])
        self.paramnames.append(self.streams[i].name + "/weights")
        self.paramdict.append({'param':self.params[-1], 'name', self.paramnames[-1],
                               'stream':i, 'type':PARAMETER_TYPE_COEFFICIENTS})
      if self.biases[i] is not None:
        self.params.append(self.biases[i])
        self.paramnames.append(self.streams[i].name + "/biases")
        self.paramdict.append({'param':self.params[-1], 'name', self.paramnames[-1],
                               'stream':i, 'type':PARAMETER_TYPE_OFFSETS})


#-------------------------------------------------------------------------------
  def retoutput(self):
    return self.Out

#-------------------------------------------------------------------------------

