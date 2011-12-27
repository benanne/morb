Morb: a modular RBM implementation in Theano
============================================

<center>![Morb logo](http://github.com/benanne/morb/raw/master/morblogo.png)</center>

Introduction
------------

Morb is a toolbox for building and training Restricted Boltzmann Machine models in Theano. It is intended to be modular, so that a variety of different models can be built from their elementary parts. A second goal is for it to be extensible, so that new algorithms and techniques can be plugged in easily.

The elementary parts in question are different types of **units**, which can be connected with different types of **parameters**. A schematic diagram of the architecture can be viewed below.

A unit type defines the distribution of that unit. For example, binary units are *Bernoulli* distributed. Several unit types are available, and new ones can be defined easily.

The different types of parameters form the trainable part of the model. These include *biases*, *regular weights*, *convolutional weights* and *third order weights*, amongst others. New parameter types can be defined by specifying the terms they contribute to the activations of each of the units they tie, the term they contribute to the model energy function, and the gradient of the energy function with respect to the parameters.

To train the model, one has to specify how the parameters should be updated in each step of the training process. This is possible by defining **updaters**, which can be composed. For example, one can combine a *contrastive divergence updater* with a *weight decay updater* and a *sparsity regularisation updater*. Momentum can also be applied with a *momentum updater*, which encapsulates another updater. Some updaters, like the contrastive divergence updater, calculate parameter updates from **statistics** obtained from training data.

Finally, a **trainer** is used to compile the symbolical parameter update expressions into a training function.

![Schematic diagram of Morb's RBM architecture](http://github.com/benanne/morb/raw/master/architecture.png)

Example
-------

Below is a simple example, in which an RBM with binary visibles and binary hiddens is trained on an unspecified dataset using one-step contrastive divergence (CD-1), with some weight decay.

```python
from morb import base, units, parameters, stats, updaters, trainers, monitors
import numpy
import theano.tensor as T

## define hyperparameters
learning_rate = 0.01
weight_decay = 0.02
minibatch_size = 32
epochs = 50

## load dataset
data = ...

## construct RBM model
rbm = base.RBM()

rbm.v = units.BinaryUnits(rbm) # visibles
rbm.h = units.BinaryUnits(rbm) # hiddens

rbm.W = parameters.ProdParameters(rbm, [rbm.v, rbm.h], initial_W) # weights
rbm.bv = parameters.BiasParameters(rbm, rbm.v, initial_bv) # visible bias
rbm.bh = parameters.BiasParameters(rbm, rbm.h, initial_bh) # hidden bias

## define a variable map, that maps the 'input' units to Theano variables.
initial_vmap = { rbm.v: T.matrix('v') }

## compute symbolic CD-1 statistics
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1)

## create an updater for each parameter variable
umap = {}
for variable in [rbm.W.W, rbm.bv.b, rbm.bh.b]:
    new_value = variable + learning_rate * (updaters.CDUpdater(rbm, variable, s) - decay * updaters.DecayUpdater(variable))
    umap[variable] = new_value

## monitor reconstruction cost during training
mse = monitors.reconstruction_mse(s, rbm.v)
 
## train the model
t = trainers.MinibatchTrainer(rbm, umap)
train = t.compile_function(initial_vmap, mb_size=minibatch_size, monitors=[mse])

for epoch in range(epochs):
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % numpy.mean(costs)
```

Disclaimer
----------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
