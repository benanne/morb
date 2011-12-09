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
print ">> Generating dataset..."
data = generate_data(1000) # np.random.randint(2, size=(10000, n_visible))
data_context = get_context(data)

data_train = data[:-1000, :]
data_eval = data[-1000:, :]
data_context_train = data_context[:-1000, :]
data_context_eval = data_context[-1000:, :]

n_visible = data.shape[1]
n_context = data_context.shape[1]
n_hidden = 100


print ">> Constructing RBM..."
rbm = rbms.BinaryBinaryCRBM(n_visible, n_hidden, n_context)
# sc10 = stats_collectors.CDkStatsCollector(rbm, [rbm.v], k=10) # CD-10 stats collector

# calculate CD-10 stats symbolically:
# s = sc10.calculate_stats({ rbm.v: T.vector('v') })

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
sc = stats_collectors.CDkStatsCollector(rbm, input_units=[rbm.v], latent_units=[rbm.h], context_units=[rbm.x], k=1)

umap = {}
for params in rbm.params_list:
    # pu =  0.001 * (param_updaters.CDParamUpdater(params, sc) + 0.02 * param_updaters.DecayParamUpdater(params))
    pu =  0.0005 * param_updaters.CDParamUpdater(params, sc)
    umap[params] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.ReconstructionMSEMonitor(sc, rbm.v)
mce = monitors.ReconstructionCrossEntropyMonitor(sc, rbm.v)

initial_vmap = { rbm.v: T.matrix('v'), rbm.x: T.matrix('x') }

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


