import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters

import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context

# DEBUGGING

from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None


# generate data
print ">> Generating dataset..."
data = generate_data(1000) # np.random.randint(2, size=(10000, n_visible))
data_context = get_context(data, N=1) # keep the number of dimensions low

data_train = data[:-1000, :]
data_eval = data[-1000:, :]
data_context_train = data_context[:-1000, :]
data_context_eval = data_context[-1000:, :]

n_visible = data.shape[1]
n_context = data_context.shape[1]
n_hidden = 20


print ">> Constructing RBM..."
numpy_rng = np.random.RandomState(123)
initial_W = np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(n_hidden+n_visible+n_context)),
                   high  =  4*np.sqrt(6./(n_hidden+n_visible+n_context)),
                   size  =  (n_visible, n_hidden, n_context)),
                   dtype =  theano.config.floatX)
initial_bv = np.zeros(n_visible, dtype = theano.config.floatX)
initial_bh = np.zeros(n_hidden, dtype = theano.config.floatX)



rbm = morb.base.RBM()
rbm.v = units.BinaryUnits(rbm, name='v') # visibles
rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
rbm.x = units.Units(rbm, name='x') # context

rbm.W = parameters.ThirdOrderParameters(rbm, [rbm.v, rbm.h, rbm.x], theano.shared(value = initial_W, name='W'), name='W') # weights
rbm.bv = parameters.BiasParameters(rbm, rbm.v, theano.shared(value = initial_bv, name='bv'), name='bv') # visible bias
rbm.bh = parameters.BiasParameters(rbm, rbm.h, theano.shared(value = initial_bh, name='bh'), name='bh') # hidden bias

initial_vmap = { rbm.v: T.matrix('v'), rbm.x: T.matrix('x') }

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], context_units=[rbm.x], k=1, mean_field_for_gibbs=[rbm.v], mean_field_for_stats=[rbm.v])

umap = {}
for var in rbm.variables:
    pu = var + 0.0005 * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.reconstruction_mse(s, rbm.v)
mce = monitors.reconstruction_crossentropy(s, rbm.v)

# train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
train = t.compile_function(initial_vmap, mb_size=32, monitors=[m, mce], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=32, monitors=[m, mce], train=False, name='evaluate', mode=mode)

epochs = 200
print ">> Training for %d epochs..." % epochs


for epoch in range(epochs):
    costs_train = [costs for costs in train({ rbm.v: data_train, rbm.x: data_context_train })]
    costs_eval = [costs for costs in evaluate({ rbm.v: data_eval, rbm.x: data_context_eval })]
    mses_train, ces_train = zip(*costs_train)
    mses_eval, ces_eval = zip(*costs_eval)
    
    mse_train = np.mean(mses_train)
    ce_train = np.mean(ces_train)
    mse_eval = np.mean(mses_eval)
    ce_eval = np.mean(ces_eval)
    
    print "Epoch %d" % epoch
    print "training set: MSE = %.6f, CE = %.6f" % (mse_train, ce_train)
    print "validation set: MSE = %.6f, CE = %.6f" % (mse_eval, ce_eval)


