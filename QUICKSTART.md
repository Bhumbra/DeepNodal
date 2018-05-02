# Introduction

DeepNodal is a high level Python framework for TensorFlow intended for rapid construction of deep networks with
automated distribution of computations across multiple devices.

## Motivation

Deep learning libraries such as TensorFlow and Theano typically require two stages to invoke in code:

1. Graph object construction.
2. Execution of graph objects with feeded data.

The first stage is often very work intensive even for somewhat simple network configurations. Matters become much more
complicated with multiple hardware devices (e.g. GPUs) to the extent where a minor adjustment to a network design can
entail a lot of changes in code.

DeepNodal is designed for research science work rather than software engineering applications. It abstracts much of
engineering components so that work can be focussed on network design instead. This is achieved using a three stage
approach:

1. Design of network and optimisations.
2. Graph object construction.
3. Execution of graph objects with feeded data.

Although this involves an additional step, DeepNodal mostly takes care over stages 2 and 3. Network specifications are
given as lists. For example a 4-layer hidden network each with 128 output units, with a 10-unit output layer is
specified as a list:

'''python
$ mod = dn.stack() 
$ mod.set_arch([128, 128, 128, 128, 10])
'''

If the network designer wishes to third layer into two layers in parallel with concatenation of their outputs to provide
the inputs to the next hidden_layer, then the only change in code required is:

'''python
$ mod = dn.stack() 
$ mod.set_arch([128, 128, (128, 128), 128, 10])
'''

This illustrates the motivation behind DeepNodal. If a change in design is conceptually simple, DeepNodal attempts to
make the corresponding programmatic change as simple. The complexity of DeepNodal arises from its flexibility to handle
practically any architecture (including convolutional, pooling layers, and skip connections), regularisation (including
regularisation losses and dropout), and multiple training regimes (e.g. with different learning rates and/or dropout
configurations) using simple intuitive commands without distracting the network designer with software engineering
implementation issues. 

This convenience of course comes at the cost of hiding the implementation, but DeepNodal provides a flexible interface
that allows the network designer to over-rule device handling. For now however, the learning facilities of DeepNodal are
limited to supervised learning and near-supervised learning (e.g. auto-encoders). In the future, I intend to improve
DeepNodal to include support for recurrent architectures and reinforcement learning.

# Installation

Python 3 and a recent TensorFlow version for Python 3 are the only requirements. Since DeepNodal is in its infancy,
there is currently no pip installer. The working path is simply the immediately directory that contains this
QUICKSTART.md file. As long as this directory is the current directly within a shell, a simple DeepNodal
example can be run from the current path using python3:

'''
$ python3 examples/mnist_softmax.py
'''

# Examples

## Softmax regression

Looking at examples/mnist_softmax.py, a simple example of softmax regression on MNIST data, the critical lines are:

'''python
input_dims = [28, 28, 1]
arch = 10
transfn = 'softmax'

  mod = dn.stream()
  mod.set_arch(arch)
  mod.set_transfn(transfn)

  net = dn.network()
  net.set_subnets(mod)
  net.set_inputs(input_dims)
'''

What is a DeepNodal `stream`? It is a model comprising a single chain of transformations to data that may include a
dropout layer, an architectural layer with trainable parameters, a transfer function, and a normalisation routine (such
as batch-normalisation or local response normalisation) in _any_ order. By definition, it has a single input and single
output. Here, the stream comprises of a simple 10-unit dense layer, with a softmax transfer function.

A DeepNodal `network` is a collection of models and their associated inputs. Each of these models is considered a subnet
of the network. A subnet may be a `stream`, but it may also be a more complicated model such as a `level` or `stack`
(see later!). After running examples/mnist_softmax.py, the model and TensorFlow logs will be saved to the /tmp/dn_logs/
path. Please inspect the TensorBoard-rendered graph after running the example, and inspect the implementation which
should be very clear.

## Sigmoid parallel layers example

Consider examples/mnist_sigmoid.py, a rather contrived example of parallel layers fitting MNIST data using a mean
squared error cost function. The critical lines are:

'''python
arch = (5, 5)
transfn = 'sigmoid'

  mod = dn.level()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_opverge(True)

  sup = dn.supervisor()
  .
  .
  sup.set_costfn('mse')
'''

Here, the model is not a stream but a DeepNodal `level`. A level is a model comprising of one or more chains in
parallel. A vergence can be specified to inter-relate chains at their input or output stage. The vergence function (by
default `vergence_fn='con'` for convergence) specifies whether the inputs or outputs are concatenated or summed (where
`vergence_fn='sum'). 

In the example above, a single input is shared by two streams in parallel. The parallel architectures of the two streams
are specified by the tuple argument (5, 5). The vergence at the output stage is specified by the line
`mod.set_opverge(True)`.

By default, a DeepNodal `supervisor` adopts a mean-cross entropy (`'mce'`) cost function, but notice how here it has been
specified to be the mean-squared error in the line `sup.set_costfn('mse')`. Again, inspection of the resulting graph in
TensorBoard should be very informative in showing the relationships within the model (i.e. the stream and level) as well
as between them and the `supervisor`. While `level` objects are very important in DeepNodal, it is unlikely that they
would be used in isolation. Instead, they are much more likely to belong to a `stack` (the next step).

## Skip connection vergence example

Have a look at examples/mnist_skip.py, which uses hidden layers with a skip connection to fit MNIST data. The critical
lines are:

'''python
arch = [100, 100, 100, 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
skipcv = [None] * (len(arch)-2) + [-2, None]

  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_skipcv(skipcv)
'''

A DeepNodal `stack` is a series of levels that are connected in sequence end-to-end but may also include skip
connection vergences between outputs of non-contiguous levels. Again the vergence function may be a concatenation
('con') or summation ('sum').

The architecture of a stack is specified as a single list, the length of which determines the number of levels. Notice
how the transfer function specifications have been succinctly expressed in a similar way with a ReLU for the first three
layers and a softmax for the last. Finally the `skipcv` specification includes a -2 integer which is deemed a relative
skip connection vergence, concatenating the output of the penultimate level with the level two levels earlier in the
stack. Have a look at the graph in TensorBoard to visualise the nature of the concatenation.

Note how the architecture specification comprises a list of integers in this case with no tuples in sight although in
the example above a level archecture was expressed as a tuple. DeepNodal is able to interpret this architecture
specification as a series of single-stream levels. In effect `arch = [100, 100, 100, 10]` is equivalent to `arch =
[(100), (100), (100), (10)]`. However if a design includes multi-stream levels, then tuples are required for those
levels which comprise of more than one stream.

## Parallel split multi-layer perceptron example

An example of mixing single- and multi- stream levels is givne in examples/mnist_parallel.py. Only one line shows this:

'''python
arch = [100, (100, None, 100), 10]
'''

This architecture specification means that the second level comprises of three streams whose inputs are shared as the
ouput from the first level. Since the last level has only a single stream, DeepNodal automatically introduces an input
vergence (`con` by default) for the following level.  Notice how the first and last stream of the second level comprise
simple dense layers whereas the second stream has an architecture of `None`, which correponds to an `identity`. This
approach therefore provides an alternative way of creating skip connection vergences.

