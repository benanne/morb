import morb
from morb import rbms, stats, updaters, trainers, monitors

import theano
import theano.tensor as T

import numpy as np

import gzip, cPickle

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context, plot_data

# DEBUGGING

from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None


# load data
print ">> Loading dataset..."

f = gzip.open('datasets/mnist.pkl.gz','rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set


# TODO DEBUG
train_set_x = train_set_x[:1000]
valid_set_x = valid_set_x[:100]


n_visible = train_set_x.shape[1]
n_hidden = 100


print ">> Constructing RBM..."
rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)
initial_vmap = { rbm.v: T.matrix('v') }

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])

sparsity_targets = { rbm.h: 0.1 }


eta = 0.001 # learning rate
sparsity_cost = 0.5

umap = {}

umap[rbm.W.var] = rbm.W.var + eta * updaters.CDUpdater(rbm, rbm.W.var, s) \
            + eta * sparsity_cost * updaters.SparsityUpdater(rbm, rbm.W.var, sparsity_targets, s)
umap[rbm.bh.var] = rbm.bh.var + eta * updaters.CDUpdater(rbm, rbm.bh.var, s) \
             + eta * sparsity_cost * updaters.SparsityUpdater(rbm, rbm.bh.var, sparsity_targets, s)
umap[rbm.bv.var] = rbm.bv.var + eta * updaters.CDUpdater(rbm, rbm.bv.var, s)



print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.reconstruction_mse(s, rbm.v)
m_model = s['model'][rbm.h]


# train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
train = t.compile_function(initial_vmap, mb_size=100, monitors=[m, m_model], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=100, monitors=[m, m_model], name='evaluate', train=False, mode=mode)




def sample_evolution(start, ns=100): # start = start data
    sample = t.compile_function(initial_vmap, mb_size=1, monitors=[m_model], name='evaluate', train=False, mode=mode)
    
    data = start
    plot_data(data)
    

    while True:
        for k in range(ns):
            for x in sample({ rbm.v: data }): # draw a new sample
                data = x[0]
            
        plot_data(data)
        







# TRAINING 

epochs = 200
print ">> Training for %d epochs..." % epochs

mses_train_so_far = []
mses_valid_so_far = []
mact_train_so_far = []
mact_valid_so_far = []

for epoch in range(epochs):
    monitoring_data_train = [(cost, m_model) for cost, m_model in train({ rbm.v: train_set_x })]
    mses_train, m_model_train_list = zip(*monitoring_data_train)
    mse_train = np.mean(mses_train)
    mean_activation_train = np.mean([np.mean(m) for m in m_model_train_list])
    
    monitoring_data = [(cost, m_model) for cost, m_model in evaluate({ rbm.v: valid_set_x })]
    mses_valid, m_model_valid_list = zip(*monitoring_data)
    mse_valid = np.mean(mses_valid)
    mean_activation_valid = np.mean([np.mean(m) for m in m_model_valid_list])
    
    # plotting
    mses_train_so_far.append(mse_train)
    mses_valid_so_far.append(mse_valid)
    mact_train_so_far.append(mean_activation_train)
    mact_valid_so_far.append(mean_activation_valid)
    
    plt.figure(1)
    plt.clf()
    plt.plot(mses_train_so_far, label='train')
    plt.plot(mses_valid_so_far, label='validation')
    plt.title("MSE")
    plt.legend()
    plt.draw()
    
    plt.figure(4)
    plt.clf()
    plt.plot(mact_train_so_far, label='train')
    plt.plot(mact_valid_so_far, label='validation')
    plt.title("Mean activation of hiddens")
    plt.legend()
    plt.draw()
    
    """
    # plot some samples
    plt.figure(2)
    plt.clf()
    plt.imshow(vdata[0][0].reshape((28, 28)))
    plt.draw()
    plt.figure(3)
    plt.clf()
    plt.imshow(vmodel[0][0].reshape((28, 28)))
    plt.draw()
    """

    
    print "Epoch %d" % epoch
    print "training set: MSE = %.6f, mean hidden activation = %.6f" % (mse_train, mean_activation_train)
    print "validation set: MSE = %.6f, mean hidden activation = %.6f" % (mse_valid, mean_activation_valid)




