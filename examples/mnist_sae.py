"""
An example of a multilayer perceptron with SGD back-prop and L2 regularisation training on MNIST data
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.0001

input_dims = [784]
arch = ["1000", 500, 250, 30, 250, 500, 1000, 784]
transfn = ['relu'] * (len(arch) - 1) + ['sigmoid']
weights = 'vsi'
weights_kwds = {'transpose': True}
net_name = 'sae'
optimiser = 'adam'
optimiser_kwds = {'epsilon': 1e-3}
write_dir = '/tmp/dn_logs/'

def main():

  # INPUT DATA

  source = dn.loaders.mnist()
  source.read_data()
  source.partition()

  # SPECIFY ARCHITECTURE

  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  for i in range(len(arch)):
    j = len(arch) - i - 1
    if i < len(arch) // 2:
      mod[i].set_weights(weights)
    else:
      mod[i].set_weights(mod[j].ret_params('weights'), **weights_kwds)

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.supervisor()
  sup.set_labels(dtype = 'float32')
  sup.set_errorq('mse')
  sup.set_work(net)
  sup.set_optimiser(optimiser, **optimiser_kwds)
  sup.add_schedule(learning_rate)

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  with sup.call_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      while True:
        data = source.next_batch('train', batch_size)
        if data:
          data = data[0].reshape([batch_size, -1])
          sup.train(data, data)
        else:
          break
      data = source.next_batch('test')
      data = data[0].reshape([len(data[0]), -1])
      summary_str = sup.test(data, data)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()

