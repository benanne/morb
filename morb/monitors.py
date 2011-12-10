import theano
import theano.tensor as T


def reconstruction_mse(stats, u):
    data = stats['data'][u]
    reconstruction = stats['model'][u]
    return T.mean((data - reconstruction) ** 2)

def reconstruction_crossentropy(stats, u):
    data = stats['data'][u]
    reconstruction_linear = stats['model_linear_activation'][u]
    return T.mean(T.sum(data*T.log(T.nnet.sigmoid(reconstruction_linear)) +
                  (1 - data)*T.log(1 - T.nnet.sigmoid(reconstruction_linear)), axis=1))                          
    # without optimisation:
    # return T.mean(T.sum(data*T.log(reconstruction) + (1 - data)*T.log(reconstruction), axis=1))
    # see http://deeplearning.net/tutorial/rbm.html, below the gibbs_hvh and gibbs_vhv code for an explanation.


# TODO: remove these three, they are trivial.    
def reconstruction(stats, u):
    return stats['model'][u]
    
def data(sc, u):
    return stats['data'][u]

def energy(stats, rbm, phase='data'):
    """
    The phase argument is either 'data' or 'model'.
    """
    return rbm.energy(stats[phase]) 

# TODO: pseudo likelihood? is that feasible?


