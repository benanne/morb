import theano
import theano.tensor as T

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # veel sneller
import numpy as np

numpy_rng = np.random.RandomState(123)
theano_rng = RandomStreams(numpy_rng.randint(2**30))


## common activation functions

sigmoid = T.nnet.sigmoid

def softmax(x):
    # expected input dimensions:
    # 0 = minibatches
    # 1 = units
    # 2 = states 
    
    r = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
    # r 0 = minibatches * units
    # r 1 = states
    
    # this is the expected input for theano.nnet.softmax
    s = theano.nnet.softmax(r)
    
    # reshape back to original shape
    return s.reshape(x.shape)


## samplers

def bernoulli(a):
    p = sigmoid(a) # the parameter of a bernoulli distribution is the sigmoid of the activation
    return theano_rng.binomial(size=a.shape, n=1, p=p, dtype=theano.config.floatX) 

def bernoulli_mean(a):
    return sigmoid(a)
           

def gaussian_fixed(a, std=1.0):
    # the mean parameter of a gaussian with fixed variance is the activation.
    return theano_rng.normal(size=a.shape, avg=a, std=std, dtype=theano.config.floatX)

def gaussian_fixed_mean(a):
    return a
        

def multinomial(a):
    # 0 = minibatches
    # 1 = units
    # 2 = states    
    r = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))
    # r 0 = minibatches * units
    # r 1 = states
    # this is the expected input for theano.nnet.softmax and theano_rng.multinomial
    p = T.nnet.softmax(r)
    s = theano_rng.multinomial(n=1, pvals=p, dtype=theano.config.floatX)    
    return s.reshape(a.shape) # reshape back to original shape
    

def multinomial_with_zero(a):
    # like multinomial, but include a zero energy state (so it's possible that the outcome is all zeros.
    r = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))   
    r0 = T.concatenate([r, T.zeros_like(r)[:, 0:1]], axis=1) # add row of zeros for zero energy state
    p0 = theano.nnet.softmax(r0)
    s0 = theano_rng.multinomial(n=1, pvals=p0, dtype=theano.config.floatX)
    s = s0[:, :-1] # cut off zero state column
    return s.reshape(a.shape) # reshape to original shape
    # TODO: test this sampler


def truncated_exponential(a, maximum=1.0):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples*(1 - T.exp(-a * maximum)))


def truncated_exponential_mean(a, maximum=1.0):
    return (1 / a) + (maximum / (1 - T.exp(maximum*a)))   



