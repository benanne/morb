import theano
import theano.tensor as T


#def autoencoder(rbm, vmap, visible_units, hidden_units, context_units=[]):
#    """
#    Takes an RBM that consists only of units that implement mean field.
#    The means of these units will be treated as activations of an autoencoder.
#    
#    Note that this can only be used for autoencoders with tied weights.
#    
#    input
#    rbm: the RBM object
#    vmap: a vmap dictionary of input units instances of the RBM mapped to theano expressions.
#    visible_units: a list of input units, the autoencoder will attempt to reconstruct these
#    hidden_units: the hidden layer of the autoencoder
#    
#    context units should simply be added in the vmap, they need not be specified.
#    
#    output
#    a vmap dictionary giving the reconstructions.
#    """
#    
#    # complete units lists
#    visible_units = rbm.complete_units_list(visible_units)
#    hidden_units = rbm.complete_units_list(hidden_units)
#    
#    # complete the supplied vmap
#    vmap = rbm.complete_vmap(vmap)
#    
#    hidden_vmap = rbm.mean_field(hidden_units, vmap)
#    hidden_vmap.update(vmap) # we can just add the supplied vmap to the hidden vmap to
#    # ensure that any context units are also in the hidden vmap. We do not run the risk
#    # of 'overwriting' anything since the hiddens and the visibles are disjoint.
#    # note that the hidden vmap need not be completed, since the hidden_units list
#    # has already been completed.
#    reconstruction_vmap = rbm.mean_field(visible_units, hidden_vmap)
#    
#    return reconstruction_vmap
    
def autoencoder(rbm, v0_vmap, visible_units, hidden_units):
    """
    Implements the autoencoder objective: the log likelihood of the visibles given the hiddens,
    where the hidden values are obtained using mean field.
    """

    full_vmap = rbm.complete_vmap(v0_vmap)
    # add the conditional means for the hidden units to the vmap
    for hu in hidden_units:
        full_vmap[hu] = hu.mean_field(v0_vmap)

    # add any missing proxies of the hiddens (unlikely, but you never know)
    full_vmap = rbm.complete_vmap(full_vmap)
    
    # get log probs of all the visibles and take the mean.
    total_log_prob = sum(T.sum(T.mean(vu.log_prob(full_vmap), 0)) for vu in visible_units) # mean over the minibatch dimension
    
    return total_log_prob
    

def mean_reconstruction(rbm, v0_vmap, visible_units, hidden_units):   
    """
    Computes the mean reconstruction for a given RBM and a set of visibles and hiddens.
    E[v|h] with h = E[h|v].
    
    input
    rbm: the RBM object
    vmap: a vmap dictionary of input units instances of the RBM mapped to theano expressions.
    visible_units: a list of input units
    hidden_units: the hidden layer of the autoencoder
    
    context units should simply be added in the vmap, they need not be specified.
    
    output
    a vmap dictionary giving the reconstructions.

    NOTE: this vmap may contain more than just the requested values, because the 'visible_units'
    units list is completed with all proxies. So it's probably not a good idea to iterate over
    the output vmap.
    """
    
    # complete units lists
    visible_units = rbm.complete_units_list(visible_units)
    hidden_units = rbm.complete_units_list(hidden_units)
    
    # complete the supplied vmap
    v0_vmap = rbm.complete_vmap(v0_vmap)
    
    hidden_vmap = rbm.mean_field(hidden_units, v0_vmap)
    hidden_vmap.update(v0_vmap) # we can just add the supplied vmap to the hidden vmap to
    # ensure that any context units are also in the hidden vmap. We do not run the risk
    # of 'overwriting' anything since the hiddens and the visibles are disjoint.
    # note that the hidden vmap need not be completed, since the hidden_units list
    # has already been completed.
    reconstruction_vmap = rbm.mean_field(visible_units, hidden_vmap)
    
    return reconstruction_vmap



def mse(units_list, vmap_targets, vmap_predictions):
    """
    Computes the mean square error between two vmaps representing data
    and reconstruction.
    
    units_list: list of input units instances
    vmap_targets: vmap dictionary containing targets
    vmap_predictions: vmap dictionary containing model predictions
    """
    return sum(T.mean((vmap_targets[u] - vmap_predictions[u]) ** 2) for u in units_list)


def cross_entropy(units_list, vmap_targets, vmap_predictions):
    """
    Computes the cross entropy error between two vmaps representing data
    and reconstruction.
    
    units_list: list of input units instances
    vmap_targets: vmap dictionary containing targets
    vmap_predictions: vmap dictionary containing model predictions
    """
    t, p = vmap_targets, vmap_predictions
    return sum((- t[u] * T.log(p[u]) - (1 - t[u]) * T.log(1 - p[u])) for u in units_list)

    
# TODO: add objectives:
# - contractive autoencoder penalty
# - denoising autoencoder? add some methods to 'noisify' a data vmap in different way (gaussian, truncate components, ..) see dA deep learning tutorial for some code
# - facilitate supervised training
