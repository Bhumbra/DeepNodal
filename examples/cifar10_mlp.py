"""
An example of a multilayer perceptron with SGD back-prop and L2 regularisation training on MNIST data
"""

import deepnodal as dn
from time import time
import datetime
import numpy as np

# PARAMETERS

n_epochs = 20
batch_size = 60
test_split = 20
lr0 = 0.01
learning_rates = {0:lr0, 5:lr0*0.1, 10:lr0*0.01, 15:lr0*0.01}

input_dims = [32, 32, 3]
arch = [100, 100, 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
dropout = [None] * (len(arch)-1) + [0.5]
reguln = 'weight_decay'
reguln_kwds = {'scale': 0.001}
normal = ['batch_norm'] + [None] * (len(arch)-1)
normal_kwds = {'decay':0.99, 'epsilon':0.1}
optimiser = 'adam'
optimiser_kwds = {'beta1':0.9, 'beta2':0.999, 'epsilon':0.001}

gcn, zca, gcn_within_depth = True, True, True
rand_flip, rand_crop = [True, False], 2
net_name = 'mlp'
write_dir = '/tmp/dn_logs/'
save_interval = 10
seed = 42

def main():
  seed = 42

  # INPUT DATA

  source = dn.loaders.cifar10()
  source.read_data(gcn=gcn, zca=zca, gcn_within_depth=gcn_within_depth)
  source.partition(seed=seed)

  # SPECIFY ARCHITECTURE

  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_dropout(dropout)
  mod.set_normal(normal, **normal_kwds)
  mod.set_reguln(reguln, **reguln_kwds)

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.supervisor()
  sup.set_optimiser(optimiser, **optimiser_kwds)
  sup.set_work(net)
  for epoch in sorted(list(learning_rates.keys())):
    index = sup.add_schedule(learning_rates[epoch])
    if index == len(learning_rates) - 1:
      sup.set_schedule(index, False) # disable dropout

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
