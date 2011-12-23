from morb.base import Updater, SumUpdater, ScaleUpdater

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
                    
    def get_update(self):
        update = self.pu.get_update() + self.momentum * self.previous_update
        self.theano_updates = { self.previous_update: update }
        return update
        
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
        positive_term = self.rbm.energy_gradient(self.variable, self.stats['data'])
        negative_term = self.rbm.energy_gradient(self.variable, self.stats['model'])
        
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
        
        return - self.rbm.energy_gradient(self.variable, vmap) # minus sign is important!
        
        

