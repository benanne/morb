import morb
from morb import rbms, stats, updaters, trainers, monitors

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
data = generate_data(200) # np.random.randint(2, size=(10000, n_visible))

n_visible = data.shape[1]
n_hidden = 100


rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)
initial_vmap = { rbm.v: T.matrix('v') }

# try to calculate weight updates using CD-1 stats
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1)

umap = {}

for var, shape in zip([rbm.W.var, rbm.bv.var, rbm.bh.var], [(rbm.n_visible, rbm.n_hidden), (rbm.n_visible,), (rbm.n_hidden,)]):
    # pu =  0.001 * (param_updaters.CDParamUpdater(params, sc) + 0.02 * param_updaters.DecayParamUpdater(params))
    pu = updaters.CDUpdater(rbm, var, s)
    pu = var + 0.0001 * updaters.MomentumUpdater(pu, 0.9, shape)
    umap[var] = pu
    

 
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.reconstruction_mse(s, rbm.v)
train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)

epochs = 50

for epoch in range(epochs):
    print "Epoch %d" % epoch
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % np.mean(costs)

