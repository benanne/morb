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

def gaussian(a, var=1.0):
    # a is the mean, var is the variance (not std or precision!)
    std = T.sqrt(var)
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
    

def exponential(a):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples)


def truncated_exponential(a, maximum=1.0):
    uniform_samples = theano_rng.uniform(size=a.shape, dtype=theano.config.floatX)
    return (-1 / a) * T.log(1 - uniform_samples*(1 - T.exp(-a * maximum)))


def truncated_exponential_mean(a, maximum=1.0):
    # return (1 / a) + (maximum / (1 - T.exp(maximum*a))) # this is very unstable around a=0, even for a=0.001 it's already problematic.
    # here is a version that switches depending on the magnitude of the input
    m_real = (1 / a) + (maximum / (1 - T.exp(maximum*a)))
    m_approx = 0.5 - (1./12)*a + (1./720)*a**3 - (1./30240)*a**5 # + (1./1209600)*a**7 # this extra term is unnecessary, it's accurate enough
    return T.switch(T.abs_(a) > 0.5, m_real, m_approx)
 


def laplacian(b, mu=0.0):
    # laplacian distributition is only exponential family when mu=0!
    uniform_samples = theano_rng.uniform(size=b.shape, dtype=theano.config.floatX)
    return mu - b*T.sgn(uniform_samples-0.5) * T.log(1 - 2*T.abs_(uniform_samples-0.5))
    
    
    
## approximate gamma sampler
# Two approximations for the gamma function are defined.
# Windschitl is very fast, but problematic close to 0, and using the reflection formula
# causes discontinuities.
# Lanczos on the other hand is extremely accurate, but slower.
   
def _log_gamma_windschitl(z):
    """
    computes log(gamma(z)) using windschitl's approximation.
    """
    return 0.5 * (T.log(2*np.pi) - T.log(z)  + z * (2 * T.log(z) - 2 + T.log(z * T.sinh(1/z) + 1 / (810*(z**6)))))
    
def _log_gamma_ratio_windschitl(z, k):
    """
    computes log(gamma(z+k)/gamma(z)) using windschitl's approximation.
    """
    return _log_gamma_windschitl(z + k) - _log_gamma_windschitl(z)
    

def _log_gamma_lanczos(z):
    # optimised by nouiz. thanks!
    assert z.dtype.startswith("float")
    # reflection formula. Normally only used for negative arguments,
    # but here it's also used for 0 < z < 0.5 to improve accuracy in
    # this region.
    flip_z = 1 - z
    # because both paths are always executed (reflected and
    # non-reflected), the reflection formula causes trouble when the
    # input argument is larger than one.
    # Note that for any z > 1, flip_z < 0.
    # To prevent these problems, we simply set all flip_z < 0 to a
    # 'dummy' value. This is not a problem, since these computations
    # are useless anyway and are discarded by the T.switch at the end
    # of the function.
    flip_z = T.switch(flip_z < 0, 1, flip_z)
    log_pi = np.asarray(np.log(np.pi), dtype=z.dtype)
    small = log_pi - T.log(T.sin(np.pi * z)) - _log_gamma_lanczos_sub(flip_z)
    big = _log_gamma_lanczos_sub(z)
    return T.switch(z < 0.5, small, big)


def _log_gamma_lanczos_sub(z): # expanded version
    # optimised by nouiz. thanks!
    # Coefficients used by the GNU Scientific Library
    # note that vectorising this function and using .sum() turns out to be
    # really slow! possibly because the dimension across which is summed is
    # really small.
    g = 7
    p = np.array([0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                  771.32342877765313, -176.61502916214059, 12.507343278686905,
                  -0.13857109526572012, 9.9843695780195716e-6,
                  1.5056327351493116e-7], dtype=z.dtype)
    z = z - 1
    x = p[0]
    for i in range(1, g + 2):
        x += p[i] / (z + i)
    t = z + g + 0.5
    pi = np.asarray(np.pi, dtype=z.dtype)
    log_sqrt_2pi = np.asarray(np.log(np.sqrt(2 * np.pi)), dtype=z.dtype)
    return log_sqrt_2pi + (z + 0.5) * T.log(t) - t + T.log(x)

    
def _log_gamma_ratio_lanczos(z, k):
    """
    computes log(gamma(z+k)/gamma(z)) using the lanczos approximation.
    """ 
    return _log_gamma_lanczos(z + k) - _log_gamma_lanczos(z)
    
 
def gamma_approx(k, theta=1):
    """
    Sample from a gamma distribution using the Wilson-Hilferty approximation.
    The gamma function itself is also approximated, so everything can be
    computed on the GPU (using the Lanczos approximation).
    """
    lmbda = 1/3.0 # according to Wilson and Hilferty
    mu = T.exp(_log_gamma_ratio_lanczos(k, lmbda))
    sigma = T.sqrt(T.exp(_log_gamma_ratio_lanczos(k, 2*lmbda)) - mu**2)
    normal_samples = theano_rng.normal(size=k.shape, avg=mu, std=sigma, dtype=theano.config.floatX)
    gamma_samples = theta * T.abs_(normal_samples ** 3)
    # The T.abs_ is technically incorrect. The problem is that, without it, this formula may yield
    # negative samples, which is impossible for the gamma distribution.
    # It was also noted that, for very small values of the shape parameter k, the distribution
    # of resulting samples is roughly symmetric around 0. By 'folding' the negative part
    # onto the positive part, we still get a decent approximation because of this.
    return gamma_samples
    
    
    
    

