""" An example of a wide resnet network with Nestorov-momentum back-prop on CIFAR10 data.  (details in Zagoruyko and Komodakis, 2017)
"""

import deepnodal as dn
from time import time
import datetime
import numpy as np

# PARAMETERS

n_epochs = 180
batch_size = 128
test_split = 50
max_lr = 0.1
learning_rates = {0:max_lr*0.2, 1:max_lr, 60:max_lr*0.2, 120:max_lr*0.04, 160:max_lr*0.008}
devs = 2 # set to None if using only one device or N to use N GPUs

input_dims = [32, 32, 3]
k, N = 10, 4
k16, k32, k64, N2 = k*16, k*32, k*64, N*2
arch = [ [ 16, [3, 3], [1, 1]] ]                        +\
       [([k16, [3, 3], [1, 1]], None)]                  +\
       [([k16, [3, 3], [1, 1]], [k16, [1, 1], [1, 1]])] +\
       [ [k16, [3, 3], [1, 1]] ] * (N2 - 2)             +\
       [ [] ]                                           +\
       [([k32, [3, 3], [2, 2]], None)]                  +\
       [([k32, [3, 3], [1, 1]], [k32, [1, 1], [2, 2]])] +\
       [ [k32, [3, 3], [1, 1]] ] * (N2 - 2)             +\
       [ [] ]                                           +\
       [([k64, [3, 3], [2, 2]], None)]                  +\
       [([k64, [3, 3], [1, 1]], [k64, [1, 1], [2, 2]])] +\
       [ [k64, [3, 3], [1, 1]] ] * (N2 - 2)             +\
       [ [[8, 8], [1, 1]], 
         10 ]
opverge = [None]                    +\
          [None, True]              +\
          [None] * (N2 - 2)         +\
          [None, None, True]        +\
          [None] * (N2 - 2)         +\
          [None, None, True]        +\
          [None] * (N2 - 2)         +\
          [None, None]            
skipcv =  [None]                    +\
          [None, None]              +\
          [None, -1] * (N - 1)      +\
          [None, None, None]        +\
          [None, -1] * (N - 1)      +\
          [None, None, None]        +\
          [None, -1] * (N - 1)      +\
          [None, None]
order =   ['ant']                   +\
          ['ant', 'a']              +\
          ['nta', 'dnta'] * (N - 1) +\
          ['nt', 'ant', 'a']        +\
          ['nta', 'dnta'] * (N - 1) +\
          ['nt', 'ant', 'a']        +\
          ['nta', 'dnta'] * (N - 1) +\
          ['nta', 'at']

transfn = ['relu'] * (len(arch)-1) + ['softmax']
kernfn = ['xcorr'] * (len(arch)-2) + ['avg', None]
dropout = 0.3
normal = 'batch_norm'
normal_kwds = {'decay':0.997, 'epsilon':1e-5}
padwin = 'same'
weights = 'vsi'
reguln = "weight_decay"
reguln_kwds = {'scale': 1e-3}
opverge_kwds = {'vergence_fn': 'sum'}
skipcv_kwds = {'vergence_fn': 'sum', 'skip_end': 'inp'}       
optimiser = 'mom'
optimiser_kwds = {'momentum': 0.9, 'use_nesterov': True}

gcn, zca, gcn_within_depth = True, False, False
rand_flip, rand_crop = [True, False], 4
net_name = 'wide_resnet_N' + str(N) + '_k' + str(k)
write_dir = '/tmp/dn_logs/'
save_interval = 10

def main(seed=42):

  # INPUT DATA

  source = dn.loaders.cifar10()
  source.read_data(gcn=gcn, zca=zca, gcn_within_depth=gcn_within_depth)
  source.partition(seed=seed)

  # SPECIFY ARCHITECTURE

  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_order(order)
  mod.set_skipcv(skipcv, **skipcv_kwds)
  mod.set_opverge(opverge, **opverge_kwds)
  mod.set_transfn(transfn)
  mod.set_kernfn(kernfn)
  mod.set_padwin(padwin)
  mod.set_dropout(dropout)
  mod.set_normal(normal, **normal_kwds)
  mod.set_weights(weights)
  mod.set_reguln(reguln, **reguln_kwds)

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.hypervisor(devs = devs)
  sup.set_optimiser(optimiser, **optimiser_kwds)
  sup.set_work(net)
  for epoch in sorted(list(learning_rates.keys())):
    index = sup.add_schedule(learning_rates[epoch])

  # CHECK FOR RESTOREPOINT

  modfiler = dn.helpers.model_filer(write_dir, net_name)
  restore_point = modfiler.interview()
  if restore_point is not None:
    seed = restore_point

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  log_out = None if write_dir is None else write_dir+net_name+"_"+now
  mod_out = None if write_dir is None else log_out + "/" + net_name
  schedule = -1
  epoch_0 = 0
  t0 = time()

  with sup.call_session(log_out, seed):
    if restore_point is not None:
      epoch_0 = int(np.ceil(float(sup.progress[1])/float(source.sets['train']['support'])))
      for i in range(epoch_0):
        if i in learning_rates:
          schedule += 1
          sup.use_schedule(schedule)
    for i in range(epoch_0, n_epochs):
      if i in learning_rates:
        schedule += 1
        sup.use_schedule(schedule)
      while True:
        data = source.next_batch('train', batch_size, \
                                 rand_flip=rand_flip, rand_crop=rand_crop)
        if not data:
          break
        sup.train(*data)
      data = source.next_batch('test')
      summary_str = sup.test(*data, split=test_split)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))
      if i and mod_out is not None:
        if not(i % save_interval) or i == n_epochs -1:
          sup.save(mod_out)

if __name__ == '__main__':
  main()
