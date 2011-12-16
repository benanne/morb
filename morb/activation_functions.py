from morb.base import activation_function

import theano.tensor as T


@activation_function
def sigmoid(x):
    return T.nnet.sigmoid(x)
    
@activation_function
def identity(x):
    return x

@activation_function
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
