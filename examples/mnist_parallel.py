"""
An example of a multilayer perceptron with a parallel split with SGD back-prop on MNIST data.
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.01

input_dims = [28, 28, 1]
arch = [100, (100, 100, None), (100, 100, None), 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']

net_name = 'mlp'
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

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.supervisor()
  sup.set_work(net)
  sup.add_schedule(learning_rate)

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  with sup.call_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      for j in range(iterations_per_epoch):
        images, labels = source.train_next_batch(batch_size)
        sup.train(images, labels)
      summary_str = sup.test(source.test_images, source.test_labels)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()

