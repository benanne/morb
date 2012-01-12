# miscellaneous utility functions

import theano
from theano import tensor
import numpy
 
    
def tensordot(a, b, axes=2):
    """
    implementation of tensordot that reduces to a regular matrix product. This allows tensordot to be GPU accelerated,
    which isn't possible with the default Theano implementation (which is just a wrapper around numpy.tensordot).
    based on code from Tijmen Tieleman's gnumpy http://www.cs.toronto.edu/~tijmen/gnumpy.html
    """
    if numpy.isscalar(axes):
        # if 'axes' is a number of axes to multiply and sum over (trailing axes
        # of a, leading axes of b), we can just reshape and use dot.         
        outshape = tensor.concatenate([a.shape[:a.ndim - axes], b.shape[axes:]])
        outndim = a.ndim + b.ndim - 2*axes
        a_reshaped = a.reshape((tensor.prod(a.shape[:a.ndim - axes]), tensor.prod(a.shape[a.ndim - axes:])))
        b_reshaped = b.reshape((tensor.prod(b.shape[:axes]), tensor.prod(b.shape[axes:])))
        return tensor.dot(a_reshaped, b_reshaped).reshape(outshape, ndim=outndim)
    elif len(axes) == 2:
        # if 'axes' is a pair of axis lists, we first shuffle the axes of a and
        # b to reduce this to the first case (note the recursion).
        a_other, b_other = tuple(axes[0]), tuple(axes[1])
        num_axes = len(a_other)
        a_order = tuple(x for x in tuple(xrange(a.ndim)) if x not in a_other) + a_other
        b_order = b_other + tuple(x for x in tuple(xrange(b.ndim)) if x not in b_other)
        a_shuffled = a.dimshuffle(a_order)
        b_shuffled = b.dimshuffle(b_order)
        return tensordot(a_shuffled, b_shuffled, num_axes)
    else:
        raise ValueError("Axes should be scalar valued or a list/tuple of len 2.")
