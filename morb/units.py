from morb.base import units_type, Units, ProxyUnits
from morb import samplers
import theano.tensor as T

# BinaryUnits = units_type(activation_functions.sigmoid, samplers.bernoulli_mf)

class BinaryUnits(Units):
    def sample(self, vmap):
        a = self.activation(vmap)
        return samplers.bernoulli(a)
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return samplers.bernoulli_mean(a)

    def free_energy_term(self, vmap):
        # softplus of unit activations, summed over # units
        s = - T.nnet.softplus(self.activation(vmap))
        # sum over all but the minibatch dimension
        return T.sum(s, axis=range(1, s.ndim))

SigmoidUnits = units_type(samplers.bernoulli_mean, samplers.bernoulli_mean)

class GaussianPrecisionProxyUnits(ProxyUnits):
    def __init__(self, rbm, units, name=None):
        func = lambda x: x**2 / 2.0
        super(GaussianPrecisionProxyUnits, self).__init__(rbm, units, func, name)
        
    def mean_field(self, vmap):
        raise NotImplementedError("No mean field for now, sorry... still have to implement this.")
        # TODO: implement E[x**2] for x gaussian
       

class GaussianUnits(Units):
    def __init__(self, rbm, name=None):
        super(GaussianUnits, self).__init__(rbm, name)
        proxy_name = (name + "_precision" if name is not None else None)
        self.precision_units = GaussianPrecisionProxyUnits(rbm, self, name=proxy_name)

    def sample(self, vmap):
        a = self.activation(vmap)
        return samplers.gaussian_fixed(a)
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return samplers.gaussian_fixed_mean(a)
        
# TODO later: gaussian units with custom fixed variance (maybe per-unit). This probably requires two proxies.

SoftmaxUnits = units_type(samplers.multinomial)

TruncatedExponentialUnits = units_type(samplers.truncated_exponential, samplers.truncated_exponential_mean)
