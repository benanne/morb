from morb.base import sampler, Sampler

import theano
import theano.tensor as T

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # veel sneller
import numpy as np

numpy_rng = np.random.RandomState(123)
theano_rng = RandomStreams(numpy_rng.randint(2**30))


@sampler
def bernoulli(a, **kwargs):
    return theano_rng.binomial(size=a.shape, n=1, p=a, dtype=theano.config.floatX) 
    
@sampler
def bernoulli_mf(a, **kwargs):
    # if sampling in CD, use mean field
    if 'cd' in kwargs and kwargs['cd'] == True:
        return a # the parameter of the bernoulli distribution is also the mean
    else:
        return bernoulli.apply(a)
        
@sampler
def bernoulli_always_mf(a, **kwargs):
    # this can be used for bernoulli visibles that are actually used to model continuous data in [0,1].
    # WARNING: NEVER USE THIS FOR HIDDENS. Use bernoulli_mf instead.
    return a
    
gaussian_always_mf = bernoulli_always_mf # this comes down to the same thing;
# for both the bernoulli and the gaussian distribution, the parameter is the mean, so
# working with the mean comes down to using the parameter directly instead of sampling.

class GaussianSampler(Sampler):
    def __init__(self, std=1.0):
        super(Sampler, self).__init__()
        self.std = std
        
    def apply(self, a, **kwargs):
        return theano_rng.normal(size=a.shape, avg=a, std=self.std, dtype=theano.config.floatX)
        
class GaussianMfSampler(Sampler):
    def __init__(self, std=1.0):
        super(Sampler, self).__init__()
        self.g = GaussianSampler(std)
        
    def apply(self, a, **kwargs):
        if 'cd' in kwargs and kwargs['cd'] == True:
            return a
        else:
            return self.g.apply(a)

gaussian = GaussianSampler(std=1.0)
gaussian_mf = GaussianMfSampler(std=1.0)
# TODO: test gaussian and gaussian_mf       


@sampler
def multinomial(a, **kwargs):
    # 0 = minibatches
    # 1 = units
    # 2 = states
    r = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))   
    
    # verwachte input theano_rng.multinomial:
    # 0 = units + minibatches
    # 1 = states
    s = self.theano_rng.multinomial(n=1, pvals=r, dtype=theano.config.floatX)
    
    return s.reshape(a.shape)
    

@sampler
def multinomial_with_zero(a):
    # like multinomial, but include a zero energy state (so it's possible that the outcome is all zeros.
    r = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))   
    r0 = T.concatenate([r, T.zeros_like(r)[:, 0:1]], axis=1) # add row of zeros for zero state
    s0 = self.theano_rng.multinomial(n=1, pvals=r0, dtype=theano.config.floatX)
    s = s0[:, :-1] # cut off zero state column
    return s.reshape(a.shape) # reshape to original shape
    # TODO: test this sampler






def tmin(a, b):
    return T.switch(a < b, a, b)
    
def tmax(a, b):
    return T.switch(a > b, a, b)
    
@sampler
def beta(a, b): # TODO LATER: MOET MET TENSOREN KUNNEN WERKEN
    # best op basis van een gamma_sampler doen
    # TODO LATER
    pass
