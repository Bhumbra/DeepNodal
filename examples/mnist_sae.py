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

input_dims = [28, 28, 1]
arch = [1000, 500, 250, 30, 250, 500, 1000, 784]
transfn = ['relu'] * (len(arch) - 1) + ['sigmoid']
weights = 'vsi'
weights_kwds = {'transpose': True}
net_name = 'sae'
optimiser = 'adam'
optimiser_kwds = {'epsilon': 1e-3}
write_dir = '/tmp/dn_logs/'

def main():

  # INPUT DATA

  source = dn.helpers.mnist()
  source.read_data()
  iterations_per_epoch = source.train_num_examples // batch_size

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
  sup.new_regime(learning_rate)

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  write_out = None if write_dir is None else write_dir+net_name+"_"+now
  with sup.call_session(write_out):
    for i in range(n_epochs):
      for j in range(iterations_per_epoch):
        images, labels = source.train_next_batch(batch_size)
        sup.train(images, images.reshape([batch_size, -1]))
      summary_str = sup.test(source.test_images, source.test_images.reshape([source.test_num_examples, -1]))
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()

