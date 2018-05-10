"""
An example of a wide resnet network with Nestorov-momentum back-prop on CIFAR10 data.
(details in Zagoruyko and Komodakis, 2017)
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 150
batch_size = 120
lr0 = 0.01
learning_rates = {0:lr0, 60:lr0*0.2, 120:lr0*0.04}
devs = 2 # set to None if using only one devices

input_dims = [32, 32, 3]
k, N = 4, 4
k16, k32, k64, N2 = k*16, k*32, k*64, N*2

arch = [ [ 16, [3, 3], [1, 1]] ]                        +\
       [ [k16, [3, 3], [1, 1]] ] * N2                   +\
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

opverge = [None]                   +\
          [None] * N2              +\
          [None, None, True]       +\
          [None] * (N2 - 2)        +\
          [None, None, True]       +\
          [None] * (N2 - 2)        +\
          [None, None]            
skipcv =  [None]                   +\
          [None, None]             +\
          [None, -1] * (N - 1)     +\
          [None, None, None]       +\
          [None, -1] * (N - 1)     +\
          [None, None, None]       +\
          [None, -1] * (N - 1)     +\
          [None, None]
order =   ['ant']                  +\
          ['ant', 'a']             +\
          ['nta', 'nta'] * (N - 1) +\
          ['nt', 'ant', 'a']       +\
          ['nta', 'nta'] * (N - 1) +\
          ['nt', 'ant', 'a']       +\
          ['nta', 'nta'] * (N - 1) +\
          ['nta', 'dat']

transfn = ['relu'] * (len(arch)-1) + ['softmax']
kernfn = ['xcorr'] * (len(arch)-2) + ['avg', None]
normal = 'batch_norm'
normal_kwds = {'decay':0.997, 'epsilon':1e-5}
padwin = 'same'
weights = 'vsi'
reguln = 2
reguln_kwds = {'scale': 2e-4}
opverge_kwds = {'vergence_fn': 'sum'}
skipcv_kwds = {'vergence_fn': 'sum', 'skip_end': 'inp'}       
optimiser = 'mom'
optimiser_kwds = {'momentum': 0.9, 'use_nesterov': True}

gcn, zca, gcn_within_depth = True, False, False
rand_horz_flip, rand_bord_crop = True, True
net_name = 'wide_resnet_' + str(N) + '_' + str(k)
write_dir = '/tmp/dn_logs/'
save_interval = 10

def main():

  # INPUT DATA

  source = dn.helpers.cifar10()
  source.read_data(gcn, zca, gcn_within_depth)
  iterations_per_epoch = source.train_num_examples // batch_size

  # SPECIFY ARCHITECTURE

  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_order(order)
  mod.set_skipcv(skipcv, **skipcv_kwds)
  mod.set_opverge(opverge, **opverge_kwds)
  mod.set_transfn(transfn)
  mod.set_kernfn(kernfn)
  mod.set_padwin(padwin)
  mod.set_normal(normal, **normal_kwds)
  mod.set_weights(weights)

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)
  net.set_reguln(reguln, **reguln_kwds)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.hypervisor(devs = devs)
  sup.set_optimiser(optimiser, **optimiser_kwds)
  sup.set_work(net)
  for i in sorted(list(learning_rates)):
    sup.new_regime(learning_rates[i])

  # CHECK FOR RESTOREPOINT

  modfiler = dn.helpers.model_filer(write_dir, net_name)
  restore_point = modfiler.interview()

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  log_out = None if write_dir is None else write_dir+net_name+"_"+now
  mod_out = None if write_dir is None else log_out + "/" + net_name
  regime = -1
  epoch_0 = 0

  with sup.new_session(log_out, restore_point):
    if restore_point is not None:
      epoch_0 = int(np.ceil(float(sup.progress[1])/float(source.train_num_examples)))
      for i in range(epoch_0):
        if i in learning_rates:
          regime += 1
          sup.use_regime(regime)
    for i in range(epoch_0, n_epochs):
      if i in learning_rates:
        regime += 1
        sup.use_regime(regime)
      for j in range(iterations_per_epoch):
        images, labels = source.train_next_batch(batch_size, rand_horz_flip, rand_bord_crop)
        sup.train(images, labels)
      summary_str = sup.test(source.test_images, source.test_labels)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))
      if i and mod_out is not None:
        if not(i % save_interval) or i == n_epochs -1:
          sup.save(mod_out)

if __name__ == '__main__':
  main()

