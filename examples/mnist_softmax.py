"""
An example of a single softmax layer regression on MNIST data.
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

n_epochs = 20
batch_size = 60
learning_rate = 0.01

input_dims = [28, 28, 1]
arch = 10
transfer_function = 'softmax'

net_name = 'softmax_layer'
write_dir = '/tmp/dn_logs/'

# INPUT DATA

source = dn.helpers.mnist()
source.read_data()
iterations_per_epoch = source.train_num_examples // batch_size

# SET UP NETWORK

mod = dn.stream()
mod.set_arch(arch)
mod.set_transfn(transfer_function)
net = dn.network(net_name)
net.set_subnets(mod)
net.set_inputs(input_dims)

# SET UP SUPERVISOR AND TRAINING

sup = dn.supervisor()
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

