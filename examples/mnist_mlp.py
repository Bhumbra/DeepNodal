"""
An example of a multilayer perceptron with SGD back-prop and L2 regularisation training on MNIST data
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.001

input_dims = [28, 28, 1]
arch = [100, 100, 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
dropout = [None] * (len(arch)-1) + [0.5]
reguln = 2
reguln_kwds = {'scale': 0.001}
normal = ['batch_norm'] + [None] * (len(arch)-1)
normal_kwds = {'momentum':0.99, 'epsilon':0.1}
optimiser = 'adam'
optimiser_kwds = {'beta1':0.9, 'beta2':0.999, 'epsilon':0.001}
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
  mod.set_dropout(dropout)
  mod.set_normal(normal, **normal_kwds)

  # SPECIFY NETWORK

  net = dn.network(net_name)
  net.set_subnets(mod)
  net.set_inputs(input_dims)
  net.set_reguln(reguln, **reguln_kwds)

  # SPECIFY SUPERVISOR AND TRAINING

  sup = dn.supervisor()
  sup.set_optimiser('adam', **optimiser_kwds)
  sup.set_work(net)
  sup.new_regime(learning_rate)
  sup.new_regime(0.1*learning_rate)
  sup.new_regime(0.01*learning_rate)
  sup.new_regime(0.01*learning_rate)
  sup.set_regime(3, False) # disable dropout

  # TRAIN AND TEST

  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  t0 = time()
  with sup.new_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      if i == n_epochs // 4:
        sup.use_regime(1)
      elif i == n_epochs // 2:
        sup.use_regime(2)
      elif i == 3 * n_epochs // 4:
        sup.use_regime(3)
      for j in range(iterations_per_epoch):
        images, labels = source.train_next_batch(batch_size)
        sup.train(images, labels)
      summary_str = sup.test(source.test_images, source.test_labels)
      print("".join(["Epoch {} ({} s): ", summary_str]).format(str(i), str(round(time()-t0))))

if __name__ == '__main__':
  main()

