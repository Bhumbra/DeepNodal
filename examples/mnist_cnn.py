"""
An example of a convolutional network with SGD back-prop on MNIST data.
"""

import deepnodal as dn
from time import time
import datetime

# PARAMETERS

input_dims = [28, 28, 1]
arch = [[16, [5, 5], [1, 1]], [[3, 3], [2, 2]], [16, [3, 3], [1, 1]], [[3, 3], [2, 2]], 100, 10]
transfer_fn = ['relu'] * (len(arch)-1) + ['softmax']
learning_rate = 0.01
batch_size = 60
n_epochs = 20

net_name = 'cnn'
write_dir = '/tmp/dn_logs/'

# INPUT DATA

source = dn.helpers.mnist()
source.read_data()
iterations_per_epoch = source.train_num_examples // batch_size

# SET UP NETWORK

mod = dn.stack(net_name+"/model")
mod.set_arch(arch)
mod.set_transfn(transfer_fn)
net = dn.network(net_name)
net.set_subnets(mod)
net.set_inputs(input_dims)

# SET UP SUPERVISOR AND TRAINING

sup = dn.supervisor(net_name+'/SGD')
sup.set_trainee(net)
sup.new_regimen(learning_rate)

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

