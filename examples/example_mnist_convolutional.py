import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters

import theano
import theano.tensor as T

import numpy as np

import gzip, cPickle, time

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context

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
train_set_x = train_set_x[:10000]
valid_set_x = valid_set_x[:1000]


# reshape data for convolutional RBM
train_set_x = train_set_x.reshape((train_set_x.shape[0], 1, 28, 28))
valid_set_x = valid_set_x.reshape((valid_set_x.shape[0], 1, 28, 28))
test_set_x = test_set_x.reshape((test_set_x.shape[0], 1, 28, 28))



visible_maps = 1
hidden_maps = 50 # 100 # 50
filter_height = 28 # 8
filter_width = 28 # 8
mb_size = 10 # 1



print ">> Constructing RBM..."
fan_in = visible_maps * filter_height * filter_width

"""
initial_W = numpy.asarray(
            self.numpy_rng.uniform(
                low = - numpy.sqrt(3./fan_in),
                high = numpy.sqrt(3./fan_in),
                size = self.filter_shape
            ), dtype=theano.config.floatX)
"""
numpy_rng = np.random.RandomState(123)
initial_W = np.asarray(
            numpy_rng.normal(
                0, 0.5 / np.sqrt(fan_in),
                size = (hidden_maps, visible_maps, filter_height, filter_width)
            ), dtype=theano.config.floatX)
initial_bv = np.zeros(visible_maps, dtype = theano.config.floatX)
initial_bh = np.zeros(hidden_maps, dtype = theano.config.floatX)



shape_info = {
  'hidden_maps': hidden_maps,
  'visible_maps': visible_maps,
  'filter_height': filter_height,
  'filter_width': filter_width,
  'visible_height': 28,
  'visible_width': 28,
  'mb_size': mb_size
}

# shape_info = None


# rbms.SigmoidBinaryRBM(n_visible, n_hidden)
rbm = morb.base.RBM()
rbm.v = units.BinaryUnits(rbm, name='v') # visibles
rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
rbm.W = parameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], theano.shared(value=initial_W, name='W'), name='W', shape_info=shape_info)
# one bias per map (so shared across width and height):
rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, theano.shared(value=initial_bv, name='bv'), name='bv')
rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, theano.shared(value=initial_bh, name='bh'), name='bh')

initial_vmap = { rbm.v: T.tensor4('v') }

# try to calculate weight updates using CD-1 stats
print ">> Constructing contrastive divergence updaters..."
s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=1, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])

umap = {}
for var in rbm.variables:
    pu =  var + 0.001 * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = monitors.reconstruction_mse(s, rbm.v)
m_data = s['data'][rbm.v]
m_model = s['model'][rbm.v]
e_data = rbm.energy(s['data']).mean()
e_model = rbm.energy(s['model']).mean()


# train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, e_data, e_model], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, m_data, m_model, e_data, e_model], name='evaluate', train=False, mode=mode)






def plot_data(d):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((28,28)), interpolation='gaussian')
    plt.draw()


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
edata_train_so_far = []
emodel_train_so_far = []
edata_so_far = []
emodel_so_far = []

start_time = time.time()

for epoch in range(epochs):
    monitoring_data_train = [(cost, energy_data, energy_model) for cost, energy_data, energy_model in train({ rbm.v: train_set_x })]
    mses_train, edata_train_list, emodel_train_list = zip(*monitoring_data_train)
    mse_train = np.mean(mses_train)
    edata_train = np.mean(edata_train_list)
    emodel_train = np.mean(emodel_train_list)
    
    monitoring_data = [(cost, data, model, energy_data, energy_model) for cost, data, model, energy_data, energy_model in evaluate({ rbm.v: valid_set_x })]
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
    print "Time: %.2f s" % (time.time() - start_time)




