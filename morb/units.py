from morb.base import units_type, Units
from morb import samplers, activation_functions

BinaryUnits = units_type(activation_functions.sigmoid, samplers.bernoulli_mf)

SigmoidUnits = units_type(activation_functions.sigmoid, samplers.bernoulli_always_mf)

GaussianUnits = units_type(activation_functions.identity, samplers.gaussian_always_mf)
# TODO: this is always mf
# also need a version that actually samples.

SoftmaxUnits = units_type(activation_functions.softmax, samplers.multinomial)

class SymmetricBetaUnits(Units): # TODO
    # Symmetric because the hiddens switch between two beta distributions, not because
    # the parameters of the distribution are chosen to be equal (this is not the case).
    pass
    
class ReLUUnits(Units): # TODO
    pass
