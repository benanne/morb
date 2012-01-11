# miscellaneous utility functions

import theano
import theano.tensor as T
from numbers import Number


# based on code from Tijmen Tieleman's gnumpy http://www.cs.toronto.edu/~tijmen/gnumpy.html
def reshape_2d(arr, n):
    """
    reshapes to 2 axes. The first n axes of the array become the first axis of the returned value. The remaining ones form the second axis.
    based on code from Tijmen Tieleman's gnumpy http://www.cs.toronto.edu/~tijmen/gnumpy.html
    """
    if n < 0: n += arr.ndim
    return arr.reshape((T.prod(arr.shape[:n]), T.prod(arr.shape[n:])))

def tensordot(a, b, axes=2):
    """
    implementation of tensordot that reduces to a regular matrix product. This allows tensordot to be GPU accelerated,
    which isn't possible with the default Theano implementation (which is just a wrapper around numpy.tensordot).
    based on code from Tijmen Tieleman's gnumpy http://www.cs.toronto.edu/~tijmen/gnumpy.html
    """
    # if 'axes' is a number of axes to multiply and sum over (trailing axes of a, leading axes of b), we can just reshape and use dot.
    if isinstance(axes, Number):
        outshape = T.concatenate([a.shape[:a.ndim - axes], b.shape[axes:]])
        outndim = a.ndim + b.ndim - 2*axes
        return T.dot(reshape_2d(a, a.ndim - axes), reshape_2d(b, axes)).reshape(outshape, ndim=outndim)
    # if 'axes' is a pair of axis lists, we first shuffle the axes of a and b to reduce this to the first case (note the recursion).
    a_other, b_other = tuple(axes[0]), tuple(axes[1])
    num_axes = len(a_other)
    a_order = filter(lambda x: x not in a_other, tuple(xrange(a.ndim))) + a_other
    b_order = b_other + filter(lambda x: x not in b_other, tuple(xrange(b.ndim)))
    a_shuffled = a.dimshuffle(a_order)
    b_shuffled = b.dimshuffle(b_order)
    return tensordot(a_shuffled, b_shuffled, num_axes)
