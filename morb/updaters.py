from morb.base import Updater, SumUpdater, ScaleUpdater
import samplers

import theano
import theano.tensor as T
import numpy as np
        

class SelfUpdater(Updater):
    def get_update(self):
        return self.variable

DecayUpdater = SelfUpdater
# weight decay: the update == the parameter values themselves
# (decay constant is taken care of by ScaleUpdater)       

class MomentumUpdater(Updater):
    def __init__(self, pu, momentum, variable_shape):
        # IMPORTANT: since this Updater has state, it requires the shape of the parameter
        # variable to be supplied at initialisation.
        super(MomentumUpdater, self).__init__(pu.variable, pu.stats_list)
        self.pu = pu
        self.momentum = momentum
        self.variable_shape = variable_shape
        
        name = pu.variable.name + "_momentum"
        self.previous_update = theano.shared(value = np.zeros(self.variable_shape, dtype=theano.config.floatX), name=name)
        
        # Update calculation has to happen in __init__, because else if get_theano_updates
        # is called before get_update, the state update will not be included in the dict.
        # This is a bit nasty, and it probably applies for all updaters with state.
        # maybe this is a design flaw?
        self.update = self.pu.get_update() + self.momentum * self.previous_update
        self.theano_updates = { self.previous_update: self.update }
                    
    def get_update(self):
        return self.update
        
    def get_theano_updates(self):
        u = self.theano_updates.copy() # the MomentumUpdater's own state updates
        u.update(self.pu.get_theano_updates()) # the state updates of the contained Updater
        return u


class CDUpdater(Updater):
    def __init__(self, rbm, variable, stats):
        super(CDUpdater, self).__init__(variable, [stats])
        # this updater has only one stats object, so make it more conveniently accessible
        self.stats = stats
        self.rbm = rbm
        
    def get_update(self):
        positive_term = self.rbm.energy_gradient_sum(self.variable, self.stats['data'])
        negative_term = self.rbm.energy_gradient_sum(self.variable, self.stats['model'])
        
        return positive_term - negative_term
                
    
class SparsityUpdater(Updater):
    def __init__(self, rbm, variable, sparsity_targets, stats):
        # sparsity_targets is a dict mapping Units instances to their target activations
        super(SparsityUpdater, self).__init__(variable, [stats])
        self.stats = stats
        self.rbm = rbm
        self.sparsity_targets = sparsity_targets
        
    def get_update(self):        
        # modify vmap: subtract target values
        # this follows the formulation in 'Biasing RBMs to manipulate latent selectivity and sparsity' by Goh, Thome and Cord (2010), formulas (8) and (9).
        vmap = self.stats['data'].copy()
        for u, target in self.sparsity_targets.items():
            if u in vmap:
                vmap[u] -= target
        
        return - self.rbm.energy_gradient_sum(self.variable, vmap) # minus sign is important!
        

class BoundUpdater(Updater):
    """
    Forces the parameter to be larger than (default) or smaller than a given value.
    When type='lower', the bound is a lower bound. This is the default behaviour.
    When type='upper', the bound is an upper bound.
    The value of the bound is 0 by default, so if no extra arguments are supplied,
    this updater will force the parameter values to be positive.
    The bound is always inclusive.
    """
    def __init__(self, pu, bound=0, type='lower'):  
        super(BoundUpdater, self).__init__(pu.variable, pu.stats_list)
        self.pu = pu
        self.bound = bound
        self.type = type
                    
    def get_update(self):
        update = self.pu.get_update()
        if self.type == 'lower':
            condition = update >= self.bound
        else: # type is 'upper'
            condition = update <= self.bound
        # return T.switch(condition, update, T.ones_like(update) * self.bound)
        return T.switch(condition, update, self.variable)
      
    def get_theano_updates(self):
        # The BoundUpdater has no state, so the only updates that should be returned
        # are those of the encapsulated updater.
        return self.pu.get_theano_updates()
        
        
        
class GradientUpdater(Updater):
    """
    Takes any objective in the form of a scalar Theano expression and uses T.grad
    to compute the update with respect to the given parameter variable.
    
    This can be used to train/finetune a model supervisedly or as an auto-
    encoder, for example.
    """
    def __init__(self, objective, variable, theano_updates={}):
        """
        the theano_updates argument can be used to pass in updates if the objective
        contains a scan op or something.
        """
        super(GradientUpdater, self).__init__(variable)
        self.update = T.grad(objective, variable)
        self.theano_updates = theano_updates
        
    def get_update(self):
        return self.update
        
    def get_theano_updates(self):
        return self.theano_updates



# class DenoisingScoreMatchingUpdater(Updater):
#     """
#     implements the denoising version of the score matching objective, an alternative to
#     maximum likelihood that doesn't require an approximation of the partition function.

#     This version uses a Gaussian kernel. Furthermore, it adds the scale factor 1/sigma**2
#     to the free energy of the model, as described in "A connection between score matching
#     and denoising autoencoders" by Vincent et al., such that it yields the denoising
#     autoencoder objective for a Gaussian-Bernoulli RBM.
    
#     This approach is only valid if the domain of the input is the real numbers. That means it
#     won't work for binary input units, or other unit types that don't define a distribution
#     on the entire real line. In practice, this is used almost exclusively with Gaussian
#     visible units.

#     std: noise level
#     """
#     def __init__(self, rbm, visible_units, hidden_units, v0_vmap, std):
#         noise_map = {}
#         noisy_vmap = {}
#         for vu in visible_units:
#             noise_map[vu] = samplers.theano_rng.normal(size=v0_vmap[vu].shape, avg=0.0, std=std, dtype=theano.config.floatX)
#             noisy_vmap[vu] = v0_vmap[vu] + noise_map[vu]

#         free_energy = rbm.free_energy(hidden_units, noisy_vmap)
#         scores = [T.grad(free_energy, noisy_vmap[u]) for u in visible_units]
#         score_map = dict(zip(visible_units, scores))

#         terms = []
#         for vu in visible_units:
#             terms.append(T.sum(T.mean((score_map[vu] + noise_map[vu]) ** 2, 0))) # mean over minibatches

#         self.update = sum(terms)

#     def get_update(self):
#         return self.update


