from morb.base import RBM
from morb import units, parameters

import theano
import theano.tensor as T

import numpy as np


### RBMS ###

class BinaryBinaryRBM(RBM): # the basic RBM, with binary visibles and binary hiddens
    def __init__(self, n_visible, n_hidden):
        super(BinaryBinaryRBM, self).__init__()
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.BinaryUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        self.W = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='W'), name='W') # weights
        self.bv = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bv(), name='bv'), name='bv') # visible bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bh(), name='bh'), name='bh') # hidden bias
        
    def _initial_W(self):
        return np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   high  =  4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   size  =  (self.n_visible, self.n_hidden)),
                   dtype =  theano.config.floatX)
        
    def _initial_bv(self):
        return np.zeros(self.n_visible, dtype = theano.config.floatX)
        
    def _initial_bh(self):
        return np.zeros(self.n_hidden, dtype = theano.config.floatX)
        
        
class SigmoidBinaryRBM(BinaryBinaryRBM):
    def __init__(self, n_visible, n_hidden):
        super(BinaryBinaryRBM, self).__init__() # skip binarybinaryRBM constructor, since we've overridden it.
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.SigmoidUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        self.W = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='W'), name='W') # weights
        self.bv = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bv(), name='bv'), name='bv') # visible bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bh(), name='bh'), name='bh') # hidden bias
        


class BinaryBinaryCRBM(BinaryBinaryRBM):
    def __init__(self, n_visible, n_hidden, n_context):
        super(BinaryBinaryCRBM, self).__init__(n_visible, n_hidden)
        # data shape
        self.n_context = n_context
        # units
        self.x = units.Units(self, name='x') # context
        # parameters
        self.A = parameters.ProdParameters(self, [self.x, self.v], theano.shared(value = self._initial_A(), name='A'), name='A') # context-to-visible weights
        self.B = parameters.ProdParameters(self, [self.x, self.h], theano.shared(value = self._initial_B(), name='B'), name='B') # context-to-hidden weights

    def _initial_A(self):
        return np.zeros((self.n_context, self.n_visible), dtype = theano.config.floatX)

    def _initial_B(self):
        return np.zeros((self.n_context, self.n_hidden), dtype = theano.config.floatX)



class GaussianBinaryRBM(RBM): # Gaussian visible units
    def __init__(self, n_visible, n_hidden):
        super(GaussianBinaryRBM, self).__init__()
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.GaussianUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        parameters.FixedBiasParameters(self, self.v.precision_units)
        self.W = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='W'), name='W') # weights
        self.bv = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bv(), name='bv'), name='bv') # visible bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bh(), name='bh'), name='bh') # hidden bias
        
    def _initial_W(self):
        return np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   high  =  4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   size  =  (self.n_visible, self.n_hidden)),
                   dtype =  theano.config.floatX)
        
    def _initial_bv(self):
        return np.zeros(self.n_visible, dtype = theano.config.floatX)
        
    def _initial_bh(self):
        return np.zeros(self.n_hidden, dtype = theano.config.floatX)


class LearntPrecisionGaussianBinaryRBM(RBM):
    """
    Important: Wp and bvp should be constrained to be negative.
    """
    def __init__(self, n_visible, n_hidden):
        super(LearntPrecisionGaussianBinaryRBM, self).__init__()
        # data shape
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        # units
        self.v = units.LearntPrecisionGaussianUnits(self, name='v') # visibles
        self.h = units.BinaryUnits(self, name='h') # hiddens
        # parameters
        self.Wm = parameters.ProdParameters(self, [self.v, self.h], theano.shared(value = self._initial_W(), name='Wm'), name='Wm') # weights
        self.Wp = parameters.ProdParameters(self, [self.v.precision_units, self.h], theano.shared(value = -np.abs(self._initial_W()), name='Wp'), name='Wp') # weights
        self.bvm = parameters.BiasParameters(self, self.v, theano.shared(value = self._initial_bias(self.n_visible), name='bvm'), name='bvm') # visible bias
        self.bvp = parameters.BiasParameters(self, self.v.precision_units, theano.shared(value = self._initial_bias(self.n_visible), name='bvp'), name='bvp') # precision bias
        self.bh = parameters.BiasParameters(self, self.h, theano.shared(value = self._initial_bias(self.n_hidden), name='bh'), name='bh') # hidden bias
        
    def _initial_W(self):
        return np.asarray( np.random.uniform(
                   low   = -4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   high  =  4*np.sqrt(6./(self.n_hidden+self.n_visible)),
                   size  =  (self.n_visible, self.n_hidden)),
                   dtype =  theano.config.floatX)
        
    def _initial_bias(self, n):
        return np.zeros(n, dtype = theano.config.floatX)


