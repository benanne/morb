from morb.base import ParamUpdater
from morb.base import SumParamUpdater
from morb.base import ScaleParamUpdater

import theano
import theano.tensor as T
import numpy as np

### PARAM UPDATERS ###

class DecayParamUpdater(ParamUpdater):
    def get_update(self):
        return self.parameters.variables
        # weight decay: the update == the parameter values themselves
        # (decay constant is taken care of by ScaleParamUpdater)
        

class MomentumParamUpdater(ParamUpdater):
    def __init__(self, pu, momentum, variable_shapes):
        # IMPORTANT: since this ParamUpdater has state, it requires the shape of the parameter
        # variables to be supplied at initialisation.
        super(MomentumParamUpdater, self).__init__(pu.parameters, pu.stats_list)
        self.pu = pu
        self.momentum = momentum
        self.variable_shapes = variable_shapes
        self.previous_update_vars = []
        
        for v, shape in zip(pu.parameters.variables, self.variable_shapes):
            name = v.name + "_momentum"
            self.previous_update_vars.append(theano.shared(value = np.zeros(shape, dtype=theano.config.floatX), name=name))
                    
    def get_update(self):
        updates = [d + self.momentum * p for d, p in zip(self.pu.get_update(), self.previous_update_vars)]
        self.theano_updates = dict(zip(self.previous_update_vars, updates))
        return updates
        
    def get_theano_updates(self):
        u = self.theano_updates.copy() # the MomentumParamUpdater's own state updates
        u.update(self.pu.get_theano_updates()) # the state updates of the contained ParamUpdater
        return u


class CDParamUpdater(ParamUpdater):
    def __init__(self, parameters, stats):
        super(CDParamUpdater, self).__init__(parameters, [stats])
        # this updater has only one stats object, so make it more conveniently accessible
        self.stats = stats
        
    def get_update(self):               
        positive_term = self.parameters.gradient(self.stats['data'])
        negative_term = self.parameters.gradient(self.stats['model'])
        
        return [p - n for p, n in zip(positive_term, negative_term)]
                
    
class SparsityParamUpdater(ParamUpdater):
    def __init__(self, parameters, sparsity_targets, stats):
        # sparsity_targets is a dict mapping Units instances to their target activations
        super(SparsityParamUpdater, self).__init__(parameters, [stats])
        self.stats = stats
        self.sparsity_targets = sparsity_targets
        
    def get_update(self):        
        # modify vmap: subtract target values
        # this follows the formulation in 'Biasing RBMs to manipulate latent selectivity and sparsity' by Goh, Thome and Cord (2010), formulas (8) and (9).
        vmap = self.stats['data'].copy()
        for u, target in self.sparsity_targets.items():
            if u in vmap:
                vmap[u] -= target
        
        return [-p for p in self.parameters.gradient(vmap)] # minus sign is important!
        
    
class VarianceTargetParamUpdater(ParamUpdater):
    def get_update(self):
        pass # TODO LATER
        
        
    
    
"""
DecayParamUpdater(Parameters p)
  * MomentumParamUpdater(ParamUpdater pu)
  * CDParamUpdater(Parameters p, Stats s)
  * SparsityTargetParamUpdater(Parameters p, Stats s, target) # this also needs stats!
  * VarianceTargetParamUpdater(Parameters p, target) # maybe, as an exercise, since it doesn't really work anyway
  * SumParamUpdater([ParamUpdater p])
  * ScaleParamUpdater([ParamUpdater p], scaling_factor)
"""

