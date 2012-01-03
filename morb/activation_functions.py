import theano
import theano.tensor as T

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
    s = T.nnet.softmax(r)
    
    # reshape back to original shape
    return s.reshape(x.shape)

def softmax_with_zero(x):
    # expected input dimensions:
    # 0 = minibatches
    # 1 = units
    # 2 = states 
    
    r = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
    # r 0 = minibatches * units
    # r 1 = states
    r0 = T.concatenate([r, T.zeros_like(r)[:, 0:1]], axis=1) # add row of zeros for zero energy state
    
    # this is the expected input for theano.nnet.softmax
    p0 = T.nnet.softmax(r0)
    
    # reshape back to original shape, but with the added state
    return p0.reshape((x.shape[0], x.shape[1], x.shape[2] + 1))   


