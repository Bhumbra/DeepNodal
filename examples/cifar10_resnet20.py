"""
An example of a resnet network with Adam back-prop on CIFAR10 data.
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.1

input_dims = [32, 32, 3]
arch = [ [16, [3, 3], [1, 1]] ] * 1                   +\
       [ [16, [3, 3], [1, 1]] ] * 4                   +\
       [([32, [3, 3], [2, 2]], None)]                 +\
       [([32, [3, 3], [1, 1]], [32, [1, 1], [2, 2]])] +\
       [ [32, [3, 3], [1, 1]] ] * 4                   +\
       [([64, [3, 3], [2, 2]], None)]                 +\
       [([64, [3, 3], [1, 1]], [64, [1, 1], [2, 2]])] +\
       [ [64, [3, 3], [1, 1]] ] * 4                   +\
       [ [[8, 8], [1, 1]], 
         1000,
         10 ]

opverge = [None]                                   +\
          [None, None, None, None]                 +\
          [None, True, None, None, None, None] * 2 +\
          [None, None, None]

skipcv =  [None]                                   +\
          [None, -1, None, -1]                     +\
          [None, None, None, -1, None, -1] * 2     +\
          [None, None, None] 

order = ['datn'] + ['dnta']*(len(arch) - 4) + ['datn']*3
transfn = ['relu'] * (len(arch)-1) + ['softmax']
kernfn = ['xcorr'] * (len(arch)-3) + ['avg', 'None', None]
normal = [None] + ['batch_norm'] * (len(arch)-4) + [None, None, None]
normal_kwds = {'momentum':0.99, 'epsilon':0.001}
padwin = 'same'
weights = 'vsi'
opverge_kwds = {'vergence_fn': 'sum'}
skipcv_kwds = {'vergence_fn': 'sum', 'skip_end': 'inp'}       

net_name = 'resnet20_cifar10'
write_dir = '/tmp/dn_logs/'

def main():

  # INPUT DATA

  source = dn.helpers.cifar10()
  source.read_data()
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

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.hypervisor()
  #sup = dn.hypervisor(devs = 2)
  sup.set_work(net)
  sup.new_regime(learning_rate)

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  with sup.new_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      for j in range(iterations_per_epoch):
        images, labels = source.train_next_batch(batch_size)
        sup.train(images, labels)
      summary_str = sup.test(source.test_images, source.test_labels)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()

