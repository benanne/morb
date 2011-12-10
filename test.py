import morb
from morb import rbms, stats_collectors, param_updaters, trainers, monitors

import theano
import theano.tensor as T

import numpy as np
import time

import matplotlib.pyplot as plt
plt.ion()

from test_utils import generate_data, get_context


# DEBUGGING

from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None

# generate data
data = generate_data(200)

# use the predefined binary-binary RBM, which has visible units (rbm.v), hidden units (rbm.h),
# a weight matrix W connecting them (rbm.W), and visible and hidden biases (rbm.bv and rbm.bh).
n_visible = data.shape[1]
n_hidden = 100
rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)

# We use single-step contrastive divergence (CD-1) to train the RBM. For this, we can use
# the CDParamUpdater. This requires a stats collector that computes the CD-1 statistics:
sc = stats_collectors.CDkStatsCollector(rbm, input_units=[rbm.v], latent_units=[rbm.h], k=1)

# We create a ParamUpdater for each Parameters instance.
umap = {}
for params in rbm.params_list:
    pu =  0.001 * param_updaters.CDParamUpdater(params, sc) # the learning rate is 0.001
    umap[params] = pu
 
# training
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.ReconstructionMSEMonitor(sc, rbm.v)
train = t.compile_function({ rbm.v: T.matrix('v') }, mb_size=32, monitors=[m], name='train', mode=mode)

epochs = 50

start_time = time.time()
for epoch in range(epochs):
    print "Epoch %d" % epoch
    costs = [m for m in train({ rbm.v: data })]
    print "MSE = %.4f" % np.mean(costs)

print "Took %.2f seconds" % (time.time() - start_time)
