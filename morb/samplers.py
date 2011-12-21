import theano
import theano.tensor as T

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams # veel sneller
import numpy as np

numpy_rng = np.random.RandomState(123)
theano_rng = RandomStreams(numpy_rng.randint(2**30))

## samplers

def bernoulli(a):
    # a is the bernoulli parameter
    return theano_rng.binomial(size=a.shape, n=1, p=a, dtype=theano.config.floatX) 

def gaussian(a, std=1.0):
    # the mean parameter of a gaussian with fixed variance is the activation.
    return theano_rng.normal(size=a.shape, avg=a, std=std, dtype=theano.config.floatX)

        

def multinomial(a):
    # 0 = minibatches
    # 1 = units
    # 2 = states
    p = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))    
    # r 0 = minibatches * units
    # r 1 = states
    # this is the expected input for theano.nnet.softmax and theano_rng.multinomial
    s = theano_rng.multinomial(n=1, pvals=p, dtype=theano.config.floatX)    
    return s.reshape(a.shape) # reshape back to original shape
    

def truncated_exponential(a, maximum=1.0):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples*(1 - T.exp(-a * maximum)))


def truncated_exponential_mean(a, maximum=1.0):
    return (1 / a) + (maximum / (1 - T.exp(maximum*a)))   



