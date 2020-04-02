"""
An example of a multilayer perceptron with SGD back-prop and L2 regularisation training on MNIST data
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.01
epochs_per_schedule = 5

input_dims = [28, 28, 1]
arch = ["100", 100, 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
dropout = [None] + [0.3] * (len(arch)-2) + [0.5]
reguln = 2
reguln_kwds = {'scale': 0.001}
normal = [None] + ['batch_norm'] * (len(arch)-1)
#normal = None
normal_kwds = {'momentum':0.99, 'epsilon':0.001} #, 'renorm':True}
optimiser = 'adam'
optimiser_kwds = {'beta1':0.9, 'beta2':0.999, 'epsilon':0.001}
net_name = 'mlp'
write_dir = '/tmp/dn_logs/'
seed = 42

def main():

  # INPUT DATA

  source = dn.loaders.mnist()
  source.read_data()
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
  sup.add_schedule(learning_rate)
  sup.add_schedule(0.1*learning_rate)
  sup.add_schedule(0.01*learning_rate)
  index = sup.add_schedule(0.01*learning_rate)
  sup.set_schedule(index, False) # disable dropout

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  with sup.call_session(write_dir+net_name+"_"+now, seed=seed):
    for i in range(n_epochs):
      if not (i % epochs_per_schedule):
        sup.use_schedule(i // epochs_per_schedule)
      while True:
        data = source.next_batch('train', batch_size)
        if data:
          sup.train(*data)
        else:
          break
      data = source.next_batch('test')
      summary_str = sup.test(*data)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()
