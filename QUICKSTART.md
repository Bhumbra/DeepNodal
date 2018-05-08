# Introduction

DeepNodal is a high level Python framework for TensorFlow intended for rapid deep learning development with
automated distribution of computations across multiple devices.

## Motivation

Deep learning libraries such as TensorFlow and Theano typically require two stages to invoke in code:

1. Graph object construction.
2. Execution and updates of graph objects with feeded data.

The first stage can require considerable amounts of coding even for somewhat simple network configurations. Matters
become much more complicated with multiple hardware devices (e.g. GPUs) to the extent where a minor adjustment to a
network design can entail a lot of changes in code. There are very useful libraries that provide convenient wrappers
around deep learning backends, with more intuitive functions for common network configurations. However, while they
might reduce of bulk of code, the resulting code remains highly repetitive.

DeepNodal adopts a different approach. It is designed with research science work in mind rather than software
engineering applications. It abstracts much of engineering components so that the coding can be focussed on network
design instead. This is achieved by adding an additional stage to the list above:

1. Specification of network design and learning regimes.
2. Graph object construction.
3. Execution and updates of graph objects with feeded data.

At first it might seem an additional first step would require an increase in code instructions. However, by introducing
this a specification stage, DeepNodal almostly entirely takes care of stages 2 and 3. By separating network and learning
design from graph construction and execution, the code is considerably shortened and much less likely to contain errors. 

The only real learning curve for coders is short and shallow, and that is to understand how to code specifications. This
is very straightforward in DeepNodal, where for instance `stack` specifications (we will describe stacks properly later)
are made using Python lists: for example consider specifying 4-hidden layers each with 128 output units, with a 10-unit
output layer:

```python
  mod = dn.stack() 
  mod.set_arch([128, 128, 128, 128, 10])
```

Now suppose the network designer wishes to replace the third layer with another with two layers in parallel converging
their outputs to provide a concatenated input to the next hidden layer, then the only change in code required is:

```python
  mod = dn.stack() 
  mod.set_arch([128, 128, (128, 128), 128, 10])
```

This illustrates the motivation behind DeepNodal. If a change in design is conceptually simple, the corresponding
programmateic change should be as simple. DeepNodal is designed to make this possible. The complexity of DeepNodal
arises from its flexibility to specify practically any architecture (including convolutional, pooling layers, and skip
connections), regularisation (including regularisation losses and dropout), and multiple training regimes (e.g. with
different learning rates and/or dropout configurations). However, by adopting simple intuitive commands that do not
distract the coder with software engineering implementation issues, DeepNodal ensures that the code is only complex as
the network design and learning regimes.

This convenience does come at the cost of hiding the implementation, but DeepNodal provides a flexible interface that
allows the network designer to over-rule device handling. For now however, the learning facilities of DeepNodal are
limited to supervised learning and near-supervised learning (e.g. auto-encoders). In the future, DeepNodal will be
improved to include support for recurrent architectures and reinforcement learning.

# Installation

Python 3 and a recent TensorFlow version for Python 3 should be the only requirements. DeepNodal has been developed on
GNU/Linux systems and therefore might not work in Windows or MacOS. Since DeepNodal is in its infancy, there is
currently no manual or pip installer. The working path is simply the immediately directory that contains this
QUICKSTART.md file.  As long as this directory is the current directly within a shell, DeepNodal can be imported using a
single import line. 

```python
import deepnodal as dn
```

Before using the library, it is a very good idea to look at the examples first. A simple example can be run from the
current path in bash by invoking python3:

```
$ python3 examples/mnist_softmax.py
```

# Examples

The MNIST examples download and extract (as necessary) the MNIST data to /tmp/mnist/ and save their logs and models to
/tmp/dn_logs/.

## Softmax regression

The simplest example is examples/mnist_softmax.py, which performs softmax regression on MNIST data. While every line of
the code is self-explanatory, the critical lines are:

```python
...
input_dims = [28, 28, 1]
arch = 10
transfn = 'softmax'
...
  mod = dn.stream()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
...
  net = dn.network()
  net.set_subnets(mod)
  net.set_inputs(input_dims)
...
```

What is a DeepNodal `stream`? It is the simplest model in DeepNodal, comprising a single chain of transformations to
data. These transformations may include dropout, an architectural layer with trainable parameters, a transfer function,
and a normalisation routine (such as batch-normalisation or local response normalisation) in _any_ order (by default in
the order just listed). A stream has a single input tensor and single output tensor. Here, the stream comprises of a
simple 10-unit dense layer, with a softmax transfer function.

A DeepNodal `network` is a collection of inter-connected models and their associated inputs. Each of these models is
considered a subnet of the network. A subnet may be a `stream`, but it may also be a more sophisticated structure such
as a `level` or `stack` (which will be described shortly). After running examples/mnist_softmax.py, the model and
TensorFlow logs will be saved to the /tmp/dn_logs/ path. Please inspect the TensorBoard-rendered graph after running the
example, and inspect the implementation since the TensorBoard graph design is intended to make the implementation as
graphically clear as possible. Notice how DeepNodal automatically flattens the input for the dense layer.

## Sigmoid parallel layers example

Consider examples/mnist_sigmoid.py, a rather contrived example of parallel layers fitting MNIST data using a mean
squared error cost function. Here, the critical lines are:

```python
...
arch = (5, 5)
transfn = 'sigmoid'
...
  mod = dn.level()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_opverge(True)
...
  sup = dn.supervisor()
...
  sup.set_costfn('mse')
```

Here, the model is not a stream but a DeepNodal `level`. A level is a model comprising of one or more streams in
parallel. A vergence can be specified to inter-relate streams at their input or output stage. The vergence function (by
default `vergence_fn='con'` for convergence) specifies whether the inputs or outputs are concatenated or summed (where
`vergence_fn='sum'`). 

In the example above, a single input is shared by two streams in parallel. The parallel architectures of the two streams
are specified by the tuple argument (5, 5). The vergence at the output stage is specified by the line
`mod.set_opverge(True)`. If desired, the `stream` components can be accessed in code: `mod.streams[i]`, where `i` is the
index.

By default, a DeepNodal `supervisor` adopts a mean-cross entropy `('mce')` cost function, but notice how here it has been
specified to be the mean-squared error in the line `sup.set_costfn('mse')`. Again, inspection of the resulting graph in
TensorBoard should be very informative in showing the relationships within the model (i.e. the level and streams) as
well as between them and the `supervisor`. While `level` objects are very much at the heart of DeepNodal models however,
it is unlikely that they would be used in isolation. Instead, they are much more likely to belong to a `stack`, which
will be described in the next example.

## Skip connection vergence example

Have a look at examples/mnist_skip.py, which uses hidden layers with a skip connection to fit MNIST data. The critical
lines are:

```python
...
arch = [100, 100, 100, 10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
skipcv = [None] * (len(arch)-2) + [-2, None]
...
  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_skipcv(skipcv)
...
```

A DeepNodal `stack` is a series of levels that are connected in sequence end-to-end but may also include skip
connection vergences between outputs of non-contiguous levels. Again the vergence function may be a concatenation
('con') or summation ('sum'). 

The `stack` is the highest fully specifiable hierarchical DeepNodal model. A DeepNodal `network` may comprise of any
permutation of inter-connected subnets whether they are stacks, levels, or streams. While a `network` is not fully
specifiable, its corresponding subnets are. This flexibility would theoretically allow construction of any
architecture. Currently however, DeepNodal learning regimes only supports one subnet. Nevertheless `stack`
specifications can be very sophisticated and provide more than enough flexibility for almost all network designs.

The architecture of a stack is specified as a single list, the length of which determines the number of levels. Notice
in the code how the transfer function specifications have been succinctly expressed in a similar way with a ReLU for the
first three layers and a softmax for the last. Finally the `skipcv` specification includes a -2 integer at the index for
the penultimate layer. The negative sign specifies a relative skip connection vergence, concatenating the output of this
penultimate level with the level two levels earlier in the stack. Have a look at the graph in TensorBoard to visualise
the nature of the concatenation. 

If desired, the `level` components can be accessed in code: `mod.levels[i]`, where `i` is the index. Note however how
the architecture specification comprises a list of integers in this case with no tuples in sight despite in the previous
example where a level architecture is expressed as a tuple. DeepNodal is able to interpret this architecture
specification as a series of single-stream levels. In effect `arch = [100, 100, 100, 10]` is equivalent to `arch =
[(100), (100), (100), (10)]`. However if a design includes multiple parallel streams, then tuples are required for those
multi-stream levels.

## Parallel split multi-layer perceptron example

An example of inter-connecting single- and multi- stream levels is given in examples/mnist_parallel.py. This feature
is evident only in one line of the code:

```python
...
arch = [100, (100, None, 100), (100, None, 100), 10]
...
```

This architecture specification means that the second and third levels comprises of three streams, the first sharing a
common input from the first level. Since the last level has only a single stream, DeepNodal can only make sense of this
stack design by introducing an input vergence (`con` by default) for this last level. Notice how the first and third
stream of the middle levels comprise simple dense layers whereas the middle stream has an architecture of `None`, which
correponds to an `identity`. This approach therefore provides an alternative way of creating skip connection vergences
between levels.

## Multi-layer perceptron with regularisation and batch-normalisation example

Network training often include design specifications, beyond architectural complexity, as well as multiple training
regimes. In the example examples/mnist_mlp, a simple multilayer perceptron design is accompanied by regularisation (and
here not too helpful!) batch normalisation specifications as well as multiple training regimes. The relevant lines are:

```python
...
dropout = [None] * (len(arch)-1) + [0.5]
reguln = 2
reguln_kwds = {'scale': 0.001}
normal = ['batch_norm'] + [None] * (len(arch)-1)
normal_kwds = {'momentum':0.99, 'epsilon':0.1}
optimiser = 'adam'
optimiser_kwds = {'beta1':0.9, 'beta2':0.999, 'epsilon':0.001}
...
  mod.set_dropout(dropout)
  mod.set_normal(normal, **normal_kwds)
...
  net.set_reguln(reguln, **reguln_kwds)
...
  sup = dn.supervisor()
  sup.set_optimiser('adam', **optimiser_kwds)
  sup.set_work(net)
  sup.new_regime(learning_rate)
  sup.new_regime(0.1*learning_rate)
  sup.new_regime(0.01*learning_rate)
  sup.new_regime(0.01*learning_rate)
  sup.set_regime(3, False) # disable dropout
...
  with sup.new_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      if i == n_epochs // 4:
        sup.use_regime(1)
      elif i == n_epochs // 2:
        sup.use_regime(2)
      elif i == 3 * n_epochs // 4:
        sup.use_regime(3)
...
```

Since the syntax is consistent with previous examples, the code should be self-explanatory. Notice how dropout is
applied to the last layer, at the `stack` specification stage whereas L2 regularisation is specified at the `network`
stage. Like dropout, batch-normalisation is specified to the stack, in this case only affecting the first level. And
unlike previous examples, for which the default stochastic gradient descent `('sgd')` optimiser was employed by the
supervisor, here an Adam optimiser is used `('adam')`. Finally, four training regimes (indexed 0 to 3) have been created
with decreasing learning rates, with the last (3) disabling all dropout. Note how in the execution code, each of the
`sup.use_regime(i)` lines is invoked at particular epochs to switch training regimens. The default regime index is 0,
and so there is no need for an additional line for the first training regime.

## Convolutional network example

Since MNIST data comprises of images, convolutional networks are more appropriate than multilayer perceptors. An example
of such a convolutional network is provided in examples/mnist_cnn. The only evidence of convolutional and pooling layers
can be seen from the architectural specification:

```python
...
arch = [[16, [5, 5], [1, 1]], [[3, 3], [2, 2]], [16, [3, 3], [1, 1]], [[3, 3], [2, 2]], 100, 10]
...
```

Since again there are no tuples in sight, it is clear that every level comprises a single stream. A convolution stream
architecture specification takes the form `[number_of_feature_maps, [kernel_size], [stride]]` whereas for pooling layers
it is: `[[pooling_size], [stride]]`. The last two level architectures are single integers and therefore regular dense
layers as shown in previous examples. The existence of convolution and pooling layers is nowhere evident elsewhere in
the code as this is all that is sufficient for DeepNodal. If you are not convinced, please inspect the corresponding
TensorBoard graph. Notice how Deepnodal automatically flattens the output of the last pooling layer give provide an
acceptable input to the first dense layer.

## LeNet convolutional network example

While the default settings for the convolutional network example above work well, a network designer will want more
control. The example in examples/mnist_lenet constructs a LeNet-style architecture, with a few lines that give
additional specifications:

```python
...
arch = [ [6, [5, 5], [1, 1]], [[2, 2], [2, 2]], 
         [16, [5, 5], [1, 1]], [[2, 2], [2, 2]], 
         [120, [5, 5], [1, 1]], [[2, 2], [2, 2]], 
         84, 
         10]
transfn = ['relu'] * (len(arch)-1) + ['softmax']
kernfn = ['xcorr', 'avg'] * 3 + [None, None]
padwin = ['valid'] + ['same'] * (len(arch)-1)
weights = 'vsi'
...
  mod = dn.stack()
  mod.set_arch(arch)
  mod.set_transfn(transfn)
  mod.set_kernfn(kernfn)
  mod.set_padwin(padwin)
  mod.set_weights(weights)
...
```

The kernel function specification `kernfn` alternates between the convolution ('xcorr') and pooling ('avg') layers.
Despite its name the default convolution is a cross-correlation ('xcorr'). The default pooling kernel function is 
max-pooling ('max') but you can see in this example it has been changed to average pooling ('avg') for all pooling
layers. 

By default the padding window is 'same' but as you can see in the example it has been changed to 'valid' for the first
level only. Also note that padding specification is ignored for the last two levels since they are dense layers. The
`weights` specification controls the weights initialisation which here is variance-scaling initialisation ('vsi'), as
recommended by He et al. (2015). Once again, do inspect the TensorBoard graph to confirm the customisations and the
histograms to see the truncated normal-distributions in weight coefficients.

## Multiple-GPU convolutional network example

This final MNIST example, examples/mnist_lenet_multigpu.py can only be run if you have multiple GPUs. It is virtually 
identical to the previous example but with only a single change:
 
```python 
...
sup = dn.hypervisor(devs = 2) 
...
```

A `hypervisor` is a `supervisor` which does exactly the same thing but distributes the work-load onto multiple GPUs.
Each GPU performs the necessary computations for a sub-batch of size `batch_size/[number_of_gpu_devs]`. During training,
the hypervisor takes the average gradients computed across the GPUs to perform the parameter updates. This approach in
effect results in training that would be identical to what would take place for unsplit training batches. 

Although the adjustments in the code above are trivial, the complexity of implementation is considerable and this can be
seen if you look at the corresponding TensorBoard graph. Increased overheads means that using multiple devices does not
necessarily result in a speed up in computation, and indeed in this example it will probably be slower. But the
advantage of distributing computations across multiple devices is the increased freedom of constructing deeper and
complex networks without running into memory limitation problems.

