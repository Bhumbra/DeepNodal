# Introduction

DeepNodal is a high level Python framework for TensorFlow intended for rapid deep learning development with
automated distribution of computations across multiple devices.

## Motivation

Deep learning libraries such as TensorFlow and Theano typically require two stages to invoke in code:

1. Graph object construction.
2. Execution and updates of graph objects with feeded data.

The first stage can require considerable amounts of coding even for somewhat simple network configurations. Matters
become much more complicated with multiple hardware devices (e.g. GPUs) to the extent where a minor adjustment to a
network design can entail a lot of changes in code. There are helpful libraries that provide convenient wrappers around
deep learning backends, with more intuitive functions for common network configurations. However, while they might
reduce some of the syntactic bulk, the resulting code remains highly vulnerable to boilerplate verbosity.

DeepNodal adopts a different approach. A model-driven engineering methodology underlies the design of DeepNodal to
provide a deep learning framework for research science work rather than software engineering applications. It aims at
using abstract representations of concepts to encapsulate much of engineering components so that the coding can be
focussed on network design instead. This is achieved by adding an additional stage to the list above:

1. Specification of network design and learning regimes.
2. Graph object construction.
3. Execution and updates of graph objects with feeded data.

At first it might seem an additional first step would require an increase in code instructions. However, by introducing
this specification stage, DeepNodal almostly entirely takes care of stages 2 and 3. By separating network and learning
design from graph construction and execution, the code is considerably shortened and much less prone to coding errors. 

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
programmatic change should be as simple. DeepNodal is designed to make this possible. The complexity of DeepNodal
arises from its flexibility to specify practically any architecture (including convolutional, pooling layers, and skip
connections), regularisation (including regularisation losses and dropout), and multiple training regimes (e.g. with
different learning rates and/or dropout configurations). However, by adopting simple intuitive commands that do not
distract the coder with software engineering implementation issues, DeepNodal ensures that the code is only complex as
the network design and learning regimes.

This convenience does come at the cost of hiding the implementation, but DeepNodal provides a flexible interface that
allows the network designer to over-rule function and device handling. For now however, the learning facilities of
DeepNodal are limited to supervised learning and near-supervised learning (e.g. auto-encoders). In the future, DeepNodal
will be improved to include support for recurrent architectures and reinforcement learning.

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

# MNIST Examples

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
('con') or summation ('sum'). Whereas a level consists of streams in parallel, a stack comprises of levels in series.

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
arch = [100, (100, 100, None), (100, 100, None), 10]
...
```

This architecture specification means that the second and third levels comprises of three streams. The output of the
first level is shared as a common to the streams of the second. Since the last level has only a single stream, DeepNodal
can only make sense of this stack design by introducing an input vergence (`con` by default) for this last level. Notice
how the first and second stream of the middle levels comprise simple dense layers whereas the final stream has an
architecture of `None`, which effectively bypasses any transformations. This approach therefore provides an alternative
way of creating skip connection vergences between levels. 

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
normal_kwds = {'decay':0.99, 'epsilon':0.1}
optimiser = 'adam'
optimiser_kwds = {'beta1':0.9, 'beta2':0.999, 'epsilon':0.001}
...
  mod.set_dropout(dropout)
  mod.set_normal(normal, **normal_kwds)
  mod.set_reguln(reguln, **reguln_kwds)
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
  with sup.call_session(write_dir+net_name+"_"+now):
    for i in range(n_epochs):
      if i == n_epochs // 4:
        sup.use_regime(1)
      elif i == n_epochs // 2:
        sup.use_regime(2)
      elif i == 3 * n_epochs // 4:
        sup.use_regime(3)
...
```

Since the syntax is consistent with previous examples, the code should be self-explanatory. Notice how at the `stack`
specification stage, dropout is applied to the last layer whereas L2 regularisation applied to all layers. Like dropout,
batch-normalisation is specified to the stack, in this case only affecting the first level.  And unlike previous
examples, for which the default stochastic gradient descent `('sgd')` optimiser was employed by the supervisor, here an
Adam optimiser is used `('adam')`. Finally, four training regimes (indexed 0 to 3) have been created with decreasing
learning rates, with the last (3) disabling all dropout. Note how in the execution code, each of the `sup.use_regime(i)`
lines is invoked at particular epochs to switch training regimens. The default regime index is 0, and so there is no
need for an additional line for the first training regime.

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

# CIFAR Examples

The simple MNIST examples above really only scratch the surface of DeepNodal since this is a quickstart guide. Some of the
more advanced functions of DeepNodal is provided in the CIFAR examples. The CIFAR examples download and extract (as
necessary) the CIFAR data to /tmp/cifar/ and save their logs and models to /tmp/dn_logs/. Depending on your hardware,
you may be at risk of running out of GPU memory, but the examples are highly tweakable (e.g. adjusting the number of GPU
devices to use, or the complexity of architecture.

## Wide ResNet CIFAR10 example

Residual networks are conceptually simple yet many ResNet coding examples are long and far from straightforward. This
runs counter to DeepNodal philosophy, and in examples/cifar10_wide_resnet, a fully featured wide ResNet example is
provided that includes a dynamic model design, multiple learning rate regimens, and a checkpoint saver/loader. And this
is all performed by a single program file with less than 150 lines of code! Most of coding features you will be familiar
with from the examples above but there are some additional features here.

```python 
...
k, N = 6, 4
k16, k32, k64, N2 = k*16, k*32, k*64, N*2
arch = [ [ 16, [3, 3], [1, 1]] ]                        +\
       [ [k16, [3, 3], [1, 1]] ] * N2                   +\
       [ [] ]                                           +\
       [([k32, [3, 3], [2, 2]], None)]                  +\
       [([k32, [3, 3], [1, 1]], [k32, [1, 1], [2, 2]])] +\
       [ [k32, [3, 3], [1, 1]] ] * (N2 - 2)             +\
       [ [] ]                                           +\
       [([k64, [3, 3], [2, 2]], None)]                  +\
       [([k64, [3, 3], [1, 1]], [k64, [1, 1], [2, 2]])] +\
       [ [k64, [3, 3], [1, 1]] ] * (N2 - 2)             +\
       [ [[8, 8], [1, 1]], 
         10 ]
...
```

This architectural description is concise and clear. Note however the level specifications given as `[ [] ]` (which
in full notation would be `[ ([]) ]`. Previously you might remember the architecture `None` referring to an equality
resulting in a skip connection. The `[]` specification is very similar since it means `identity` to DeepNodal, but
without default exclusion of other stream operations, such as batch normalisation or a transfer function. This allows
simple specification of levels without any dense multiplication, convolution, or pooling. Since these levels are
followed by multi-stream levels it is clear how the shortcut connections are specified. Rather than leaving DeepNodal
to make sense of how combine multistream and unistream levels, we specify it manually.

```python 
...
opverge = [None]                   +\
          [None] * N2              +\
          [None, None, True]       +\
          [None] * (N2 - 2)        +\
          [None, None, True]       +\
          [None] * (N2 - 2)        +\
          [None, None]            
...
```

But that still leaves the majority of skip connections unaccounted for. The next specification provides that information:

```python 
...
skipcv =  [None]                   +\
          [None, None]             +\
          [None, -1] * (N - 1)     +\
          [None, None, None]       +\
          [None, -1] * (N - 1)     +\
          [None, None, None]       +\
          [None, -1] * (N - 1)     +\
          [None, None]
...
```
You will notice that source of each skip connection is `-1` rather than `-2`. The reason for this becomes clear if you
look at the remainder of the code relevant to skip connections:

```python 
...
opverge_kwds = {'vergence_fn': 'sum'}
skipcv_kwds = {'vergence_fn': 'sum', 'skip_end': 'inp'}       
...
  mod.set_skipcv(skipcv, **skipcv_kwds)
  mod.set_opverge(opverge, **opverge_kwds)
...
```

The `vergence_fn` has be changed from the default convergence ('con') to summation ('sum') for both output vergences and
skip connection vergences. But an additional specification for the skip connection vergences has been given as
`'skip_end': 'inp'` where by default setting is `'skip_end': 'out'`. This change means that source of the skip
connection is to be taken from _input_ rather than _output_ of the referenced level. It is for this reason that skip
connection specification given above uses `-1` rather than `-2`. It is sometimes desirable (and in this case necessary)
to do this when mixing skip connection vergences and output vergences.

As well as skip connections, a peculiar feature of wide residual networks is switching the order of convolution,
transfer function, and batch normalisation operations around. These switches are specified next:

```python 
...
order =   ['ant']                  +\
          ['ant', 'a']             +\
          ['nta', 'nta'] * (N - 1) +\
          ['nt', 'ant', 'a']       +\
          ['nta', 'nta'] * (N - 1) +\
          ['nt', 'ant', 'a']       +\
          ['nta', 'nta'] * (N - 1) +\
          ['nta', 'dat']
...          
  mod.set_order(order)
...
```

The letters 'datn' stand for 'dropout', 'architecture', 'transfer function', and 'normalisation' respectively. The order
specification is incredibly powerful because not only does it allow assignment of the order of operations but also
allows specific exclusion of operations. For example, 'a' allows only an architectural operation (here a convolution)
and excludes all other types of operations. Since ResNets do not generalise well with adaptive learning rate optimisers,
a momentum optimiser is specified:

```python 
...
optimiser = 'mom'
optimiser_kwds = {'momentum': 0.9, 'use_nesterov': True}
...
```

Another difference from previous examples, is the specification of splitting the test data.

```python 
...
test_split = 10
...
      summary_str = sup.test(source.test_images, source.test_labels, split = test_split)
...
```

By default, DeepNodal does not split up the test data into separate test batches. This default however is vulnerable to
memory overloads. Using hypervisor with `devs = num_gpus` reduces this risk my distributing the batches over separate
graphics units. However, this may still not be enough of a reduction. As a result the lines above further split the test
batch (into 10 in this case). Changing the split does not affect the summary test results, but it is the responsibility
of the coder to make sure the number of test data is divisible by `(num_gpus * test_split)`.

Finally for this example, we include the how to save and load of checkpoints.

```python 
...
save_interval = 10
...
  modfiler = dn.helpers.model_filer(write_dir, net_name)
  restore_point = modfiler.interview()
...
  now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
  log_out = None if write_dir is None else write_dir+net_name+"_"+now
  mod_out = None if write_dir is None else log_out + "/" + net_name
  regime = -1
  epoch_0 = 0
  t0 = time()

  with sup.call_session(log_out, restore_point):
    if restore_point is not None:
      epoch_0 = int(np.ceil(float(sup.progress[1])/float(source.train_num_examples)))
      for i in range(epoch_0):
        if i in learning_rates:
          regime += 1
          sup.use_regime(regime)
    for i in range(epoch_0, n_epochs):
...
      if i and mod_out is not None:
        if not(i % save_interval) or i == n_epochs -1:
          sup.save(mod_out)
...
```
From the last few lines of the code, you can see the model is saved after a defined interval of epochs (set here to 10)
and again at the end. These checkpoints can be loaded by specificying a restore point when creating a new session. For
convenience, a model filer lists checkpoints that _may_ be suitable. Note that what is offered is based solely on
filename and no checks are made to ensure that the checkpoints match the network and learning specifications. Using
checkpoints however allows training to be broken up conveniently into separate sessions.

