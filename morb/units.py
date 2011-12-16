from morb.base import units_type, Units
from morb import samplers, activation_functions
import theano.tensor as T

# BinaryUnits = units_type(activation_functions.sigmoid, samplers.bernoulli_mf)

class BinaryUnits(Units):
    def __init__(self, rbm, name=None):
        super(BinaryUnits, self).__init__(rbm, activation_functions.sigmoid, samplers.bernoulli_mf, name=name)

    def free_energy_term(self, vmap):
        # softplus of unit activations, summed over # units
        s = - T.nnet.softplus(self.linear_activation(vmap))
        # sum over all but the minibatch dimension
        return T.sum(s, axis=range(1, s.ndim))

SigmoidUnits = units_type(activation_functions.sigmoid, samplers.bernoulli_always_mf)

GaussianUnits = units_type(activation_functions.identity, samplers.gaussian_mf) # std = 1

def gaussian_units_type(std, mean_field=True):
    """
    Factory method to create gaussian units with a given standard deviation
    """
    if mean_field:
        sampler = samplers.GaussianMfSampler(std)
    else:
        sampler = samplers.GaussianSampler(std)
    return units_type(activation_functions.identity, sampler)

MeanFieldGaussianUnits = units_type(activation_functions.identity, samplers.gaussian_always_mf)

SoftmaxUnits = units_type(activation_functions.softmax, samplers.multinomial)

class SymmetricBetaUnits(Units): # TODO
    # Symmetric because the hiddens switch between two beta distributions, not because
    # the parameters of the distribution are chosen to be equal (this is not the case).
    pass
    
class ReLUUnits(Units): # TODO
    pass
