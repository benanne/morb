import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters

import theano
import theano.tensor as T

import numpy as np

import gzip, cPickle

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context, one_hot

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

# convert labels to one hot representation
train_set_y_oh = one_hot(np.atleast_2d(train_set_y).T)
valid_set_y_oh = one_hot(np.atleast_2d(valid_set_y).T)
test_set_y_oh = one_hot(np.atleast_2d(test_set_y).T)

# dim 0 = minibatches, dim 1 = units, dim 2 = states
train_set_y_oh = train_set_y_oh.reshape((train_set_y_oh.shape[0], 1, train_set_y_oh.shape[1]))
valid_set_y_oh = valid_set_y_oh.reshape((valid_set_y_oh.shape[0], 1, valid_set_y_oh.shape[1]))
test_set_y_oh = test_set_y_oh.reshape((test_set_y_oh.shape[0], 1, test_set_y_oh.shape[1]))


# make the sets a bit smaller for testing purposes
train_set_x = train_set_x[:10000]
train_set_y_oh = train_set_y_oh[:10000]
valid_set_x = valid_set_x[:1000]
valid_set_y_oh = valid_set_y_oh[:1000]




n_visible = train_set_x.shape[1]
n_hidden = 100
n_states = train_set_y_oh.shape[2]


print ">> Constructing RBM..."
rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)

# add softmax unit for context
rbm.s = units.SoftmaxUnits(rbm, name='s')

# link context and hiddens
initial_Ws = np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(n_hidden+1+n_states)),
                   high  =  4*np.sqrt(6./(n_hidden+1+n_states)),
                   size  =  (1, n_states, n_hidden)),
                   dtype =  theano.config.floatX)
rbm.Ws = parameters.AdvancedProdParameters(rbm, [rbm.s, rbm.h], [2, 1], theano.shared(value = initial_Ws, name='Ws'), name='Ws')

initial_vmap = { rbm.v: T.matrix('v'), rbm.s: T.tensor3('s') }

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], context_units=[rbm.s], k=1, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])
# s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v, rbm.s], hidden_units=[rbm.h], k=1, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])

umap = {}
for var in rbm.variables:
    pu = var + 0.001 * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.reconstruction_mse(s, rbm.v)
m_data = s['data'][rbm.v]
m_model = s['model'][rbm.v]
e_data = rbm.energy(s['data']).mean()
e_model = rbm.energy(s['model']).mean()

train = t.compile_function(initial_vmap, mb_size=100, monitors=[m, e_data, e_model], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=100, monitors=[m, m_data, m_model, e_data, e_model], name='evaluate', train=False, mode=mode)






def plot_data(d):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((28,28)), interpolation='gaussian')
    plt.draw()


def sample_evolution(start, cls, ns=100): # start = start data
    sample = t.compile_function(initial_vmap, mb_size=1, monitors=[m_model], name='evaluate', train=False, mode=mode)
    
    data = start
    plot_data(data)
    
    label = one_hot(np.atleast_2d(cls), dim=10)
    label = label.reshape((label.shape[0], 1, label.shape[1]))
    

    while True:
        for k in range(ns):
            for x in sample({ rbm.v: data, rbm.s: label }): # draw a new sample
                data = x[0]
            
        plot_data(data)
        









# TRAINING 

epochs = 200
print ">> Training for %d epochs..." % epochs

mses_train_so_far = []
mses_valid_so_far = []
edata_train_so_far = []
emodel_train_so_far = []
edata_so_far = []
emodel_so_far = []

for epoch in range(epochs):
    monitoring_data_train = [(cost, energy_data, energy_model) for cost, energy_data, energy_model in train({ rbm.v: train_set_x, rbm.s: train_set_y_oh })]
    mses_train, edata_train_list, emodel_train_list = zip(*monitoring_data_train)
    mse_train = np.mean(mses_train)
    edata_train = np.mean(edata_train_list)
    emodel_train = np.mean(emodel_train_list)
    
    monitoring_data = [(cost, data, model, energy_data, energy_model) for cost, data, model, energy_data, energy_model in evaluate({ rbm.v: valid_set_x, rbm.s: valid_set_y_oh })]
    mses_valid, vdata, vmodel, edata, emodel = zip(*monitoring_data)
    mse_valid = np.mean(mses_valid)
    edata_valid = np.mean(edata)
    emodel_valid = np.mean(emodel)
    
    # plotting
    mses_train_so_far.append(mse_train)
    mses_valid_so_far.append(mse_valid)
    edata_so_far.append(edata_valid)
    emodel_so_far.append(emodel_valid)
    edata_train_so_far.append(edata_train)
    emodel_train_so_far.append(emodel_train)
    
    plt.figure(1)
    plt.clf()
    plt.plot(mses_train_so_far, label='train')
    plt.plot(mses_valid_so_far, label='validation')
    plt.title("MSE")
    plt.legend()
    plt.draw()
    
    plt.figure(4)
    plt.clf()
    plt.plot(edata_so_far, label='validation / data')
    plt.plot(emodel_so_far, label='validation / model')
    plt.plot(edata_train_so_far, label='train / data')
    plt.plot(emodel_train_so_far, label='train / model')
    plt.title("energy")
    plt.legend()
    plt.draw()
    
    # plot some samples
    plt.figure(2)
    plt.clf()
    plt.imshow(vdata[0][0].reshape((28, 28)))
    plt.draw()
    plt.figure(3)
    plt.clf()
    plt.imshow(vmodel[0][0].reshape((28, 28)))
    plt.draw()

    
    print "Epoch %d" % epoch
    print "training set: MSE = %.6f, data energy = %.2f, model energy = %.2f" % (mse_train, edata_train, emodel_train)
    print "validation set: MSE = %.6f, data energy = %.2f, model energy = %.2f" % (mse_valid, edata_valid, emodel_valid)




