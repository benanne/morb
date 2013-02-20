import morb
from morb import rbms, stats, updaters, trainers, monitors, objectives

import theano
import theano.tensor as T

import numpy as np

import gzip, cPickle

import matplotlib.pyplot as plt
plt.ion()

from utils import generate_data, get_context, visualise_filters

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


train_set_x = train_set_x[:10000]
valid_set_x = valid_set_x[:1000]


n_visible = train_set_x.shape[1]
n_hidden = 100 # 500
mb_size = 100
k = 1 # 15
learning_rate = 0.1 # 0.1 # 0.02 # 0.1
epochs = 500
corruption_level = 0.0 # 0.3
sparsity_penalty = 0.1 # 0.0
sparsity_target = 0.05


print ">> Constructing RBM..."
# rbm = rbms.GaussianBinaryRBM(n_visible, n_hidden)
rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)
# rbm = rbms.TruncExpBinaryRBM(n_visible, n_hidden)

v = T.matrix('v')
v_corrupted = objectives.corrupt_masking(v, corruption_level)
initial_vmap = { rbm.v: v }
initial_vmap_corrupted = { rbm.v: v_corrupted }

print ">> Constructing autoencoder updaters..."
autoencoder_objective = objectives.autoencoder(rbm, [rbm.v], [rbm.h], initial_vmap, initial_vmap_corrupted)
reconstruction = objectives.mean_reconstruction(rbm, [rbm.v], [rbm.h], initial_vmap_corrupted)

autoencoder_objective += sparsity_penalty * objectives.sparsity_penalty(rbm, [rbm.h], initial_vmap, sparsity_target)

umap = {}
for var in rbm.variables:
    pu = var + (learning_rate / float(mb_size)) * updaters.GradientUpdater(autoencoder_objective, var)
    umap[var] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)
m = T.mean((initial_vmap[rbm.v] - reconstruction[rbm.v]) ** 2)
c = T.mean(T.nnet.binary_crossentropy(reconstruction[rbm.v], initial_vmap[rbm.v]))

# train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[autoencoder_objective, m, c], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[autoencoder_objective, m, c], name='evaluate', train=False, mode=mode)






def plot_data(d, *args, **kwargs):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((28,28)), interpolation='gaussian', *args, **kwargs)
    plt.draw()








# TRAINING 

print ">> Training for %d epochs..." % epochs

mses_train_so_far = []
mses_valid_so_far = []
objs_train_so_far = []
objs_valid_so_far = []
ces_train_so_far = []
ces_valid_so_far = []


for epoch in range(epochs):
    objs_train, mses_train, ces_train = zip(*train({ rbm.v: train_set_x }))
    mse_train = np.mean(mses_train)
    ce_train = np.mean(ces_train)
    obj_train = np.mean(objs_train)
    
    objs_valid, mses_valid, ces_valid = zip(*evaluate({ rbm.v: valid_set_x }))
    mse_valid = np.mean(mses_valid)
    ce_valid = np.mean(ces_valid)
    obj_valid = np.mean(objs_valid)
    
    # plotting
    mses_train_so_far.append(mse_train)
    mses_valid_so_far.append(mse_valid)
    objs_train_so_far.append(obj_train)
    objs_valid_so_far.append(obj_valid)
    ces_train_so_far.append(ce_train)
    ces_valid_so_far.append(ce_valid)
    
    plt.figure(1)
    plt.clf()
    plt.plot(mses_train_so_far, label='train')
    plt.plot(mses_valid_so_far, label='validation')
    plt.title("MSE")
    plt.legend()
    plt.draw()
    
    plt.figure(2)
    plt.clf()
    plt.plot(objs_train_so_far, label='train')
    plt.plot(objs_valid_so_far, label='validation')
    plt.title("objective")
    plt.legend()
    plt.draw()
    
    plt.figure(3)
    plt.clf()
    plt.plot(ces_train_so_far, label='train')
    plt.plot(ces_valid_so_far, label='validation')
    plt.title("cross-entropy")
    plt.legend()
    plt.draw()
    
    plt.figure(4)
    plt.clf()
    visualise_filters(rbm.W.var.get_value(), dim=28)
    plt.colorbar()
    plt.title("filters")
    plt.draw()
    
    
    print "Epoch %d" % epoch
    print "training set: MSE = %.6f, CE = %.6f, objective = %.6f" % (mse_train, ce_train, obj_train)
    print "validation set: MSE = %.6f, CE = %.6f, objective = %.6f" % (mse_valid, ce_valid, obj_valid)




