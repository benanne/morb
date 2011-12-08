import morb
from morb import rbms, stats_collectors, param_updaters, trainers, monitors

import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt
plt.ion()

from test_utils import generate_data, get_context


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
# sc10 = stats_collectors.CDkStatsCollector(rbm, [rbm.v], k=10) # CD-10 stats collector

# calculate CD-10 stats symbolically:
# s = sc10.calculate_stats({ rbm.v: T.vector('v') })

# try to calculate weight updates using CD-1 stats
sc = stats_collectors.CDkStatsCollector(rbm, input_units=[rbm.v], latent_units=[rbm.h], k=1)

umap = {}
for params in rbm.params_list:
    # pu =  0.001 * (param_updaters.CDParamUpdater(params, sc) + 0.02 * param_updaters.DecayParamUpdater(params))
    pu =  0.001 * param_updaters.CDParamUpdater(params, sc)
    umap[params] = pu

 
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.ReconstructionMSEMonitor(sc, rbm.v)
train = t.compile_function({ rbm.v: T.matrix('v') }, mb_size=32, monitors=[m], name='train', mode=mode)

epochs = 50

for epoch in range(epochs):
    print "Epoch %d" % epoch
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % np.mean(costs)

