from morb.base import Units, ProxyUnits
from morb import samplers, activation_functions
import theano.tensor as T


class BinaryUnits(Units):
    def sample(self, vmap):
        a = self.activation(vmap)
        p = activation_functions.sigmoid(a)
        return samplers.bernoulli(p)
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return activation_functions.sigmoid(a)

    def free_energy_term(self, vmap):
        # softplus of unit activations, summed over # units
        s = - T.nnet.softplus(self.activation(vmap))
        # sum over all but the minibatch dimension
        return T.sum(s, axis=range(1, s.ndim))
  
  
class GaussianPrecisionProxyUnits(ProxyUnits):
    def __init__(self, rbm, units, name=None):
        func = lambda x: x**2 / 2.0
        super(GaussianPrecisionProxyUnits, self).__init__(rbm, units, func, name)
       
class GaussianUnits(Units):
    def __init__(self, rbm, name=None):
        super(GaussianUnits, self).__init__(rbm, name)
        proxy_name = (name + "_precision" if name is not None else None)
        self.precision_units = GaussianPrecisionProxyUnits(rbm, self, name=proxy_name)

    def sample(self, vmap):
        a = self.activation(vmap)
        return samplers.gaussian(a)
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return a


class LearntPrecisionGaussianProxyUnits(ProxyUnits):
    def __init__(self, rbm, units, name=None):
        func = lambda x: x**2
        super(LearntPrecisionGaussianProxyUnits, self).__init__(rbm, units, func, name)
             
class LearntPrecisionGaussianUnits(Units):
    def __init__(self, rbm, name=None):
        super(LearntPrecisionGaussianUnits, self).__init__(rbm, name)
        proxy_name = (name + "_precision" if name is not None else None)
        self.precision_units = LearntPrecisionGaussianProxyUnits(rbm, self, name=proxy_name)
        self.proxy_units = [self.precision_units]

    def sample(self, vmap):
        a1 = self.activation(vmap)
        a2 = self.precision_units.activation(vmap)
        return samplers.gaussian(a1/(-2*a2), 1/(-2*a2))
        
       
# TODO later: gaussian units with custom fixed variance (maybe per-unit). This probably requires two proxies.

class SoftmaxUnits(Units):
    # 0 = minibatches
    # 1 = units
    # 2 = states
    def sample(self, vmap):
        a = self.activation(vmap)
        p = activation_functions.softmax(a)
        return samplers.multinomial(p)


class SoftmaxWithZeroUnits(Units):
    """
    Like SoftmaxUnits, but in this case a zero state is possible, yielding N+1 possible states in total.
    """
    def sample(self, vmap):
        a = self.activation(vmap)
        p0 = activation_functions.softmax_with_zero(a)
        s0 = samplers.multinomial(p0)
        s = s0[:, :, :-1] # chop off the last state (zero state)
        return s


class TruncatedExponentialUnits(Units):
    def sample(self, vmap):
        a = self.activation(vmap)
        return samplers.truncated_exponential(-a) # lambda = -a!
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return samplers.truncated_exponential_mean(-a)



class ExponentialUnits(Units):
    def sample(self, vmap):
        a = self.activation(vmap)
        return samplers.exponential(-a) # lambda = -a!
        
    def mean_field(self, vmap):
        a = self.activation(vmap)
        return 1.0 / (-a)
        
        

class NRELUnits(Units):
    """
    Noisy rectified linear units from 'Rectified Linear Units Improve Restricted Boltzmann Machines'
    by Nair & Hinton (ICML 2010)
    
    WARNING: computing the energy or free energy of a configuration does not have the same semantics
    as usual with NReLUs, because each ReLU is actually the sum of an infinite number of Bernoulli
    units with offset biases. The energy depends on the individual values of these Bernoulli units,
    whereas only the sum is ever sampled (approximately).
    """
    def sample(self, vmap):
        a = self.activation(vmap)
        s = a + samplers.gaussian(0, T.nnet.sigmoid(a)) # approximation: linear + gaussian noise
        return T.max(0, s) # rectify
        
    def mean_field(self, vmap):
        return T.max(0, self.activation(vmap))

