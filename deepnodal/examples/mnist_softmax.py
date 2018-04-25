"""
An example of a single softmax layer regression on MNIST data.
"""

import deepnodal as dn
from deepnodal.helpers import mnist
from deepnodal.structures import stream, network
from deepnodal.functions import supervisor

# INPUT DATA

source = mnist.mnist()
source.read_data()
#images, labels = source.train_next_batch(10)

