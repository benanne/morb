import theano
import theano.tensor as T


def reconstruction_mse(stats, u):
    data = stats['data'][u]
    reconstruction = stats['model'][u]
    return T.mean((data - reconstruction) ** 2)
    
def reconstruction_error_rate(stats, u):
    data = stats['data'][u]
    reconstruction = stats['model'][u]
    return T.mean(T.neq(data, reconstruction))

def reconstruction_crossentropy(stats, u):
    data = stats['data'][u]
    reconstruction_activation = stats['model_activation'][u]
    return T.mean(T.sum(data*T.log(T.nnet.sigmoid(reconstruction_activation)) +
                  (1 - data)*T.log(1 - T.nnet.sigmoid(reconstruction_activation)), axis=1))                          
    # without optimisation:
    # return T.mean(T.sum(data*T.log(reconstruction) + (1 - data)*T.log(reconstruction), axis=1))
    # see http://deeplearning.net/tutorial/rbm.html, below the gibbs_hvh and gibbs_vhv code for an explanation.



# TODO: pseudo likelihood? is that feasible?


