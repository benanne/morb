from morb.base import ParamUpdater
from morb.base import SumParamUpdater
from morb.base import ScaleParamUpdater

### PARAM UPDATERS ###

class DecayParamUpdater(ParamUpdater):
    def calculate_update(self):
        return self.parameters.variables
        # weight decay: the update == the parameter values themselves
        # (decay constant is taken care of by ScaleParamUpdater)
        

class MomentumParamUpdater(ParamUpdater):
    def calculate_update(self):
        pass # TODO LATER: momentum: this updater has to have memory, so it's a challenge.
    

class CDParamUpdater(ParamUpdater):
    def __init__(self, parameters, stats_collector):
        super(CDParamUpdater, self).__init__(parameters, [stats_collector])
        # this updater has only one stats collector, so make it more conveniently accessible
        self.stats_collector = stats_collector 
        
    def calculate_update(self):
        stats = self.stats_collector.stats
                
        positive_term = self.parameters.gradient(stats['data'])
        negative_term = self.parameters.gradient(stats['model'])
        
        return [p - n for p, n in zip(positive_term, negative_term)]
                
    
class SparsityParamUpdater(ParamUpdater):
    def __init__(self, parameters, sparsity_targets, stats_collector):
        # sparsity_targets is a dict mapping Units instances to their target activations
        super(SparsityParamUpdater, self).__init__(parameters, [stats_collector])
        self.stats_collector = stats_collector
        self.sparsity_targets = sparsity_targets
        
    def calculate_update(self):
        stats = self.stats_collector.stats
        
        # modify vmap: subtract target values
        # this follows the formulation in 'Biasing RBMs to manipulate latent selectivity and sparsity' by Goh, Thome and Cord (2010), formulas (8) and (9).
        vmap = stats['data'].copy()
        for u, target in self.sparsity_targets.items():
            if u in vmap:
                vmap[u] -= target
        
        return [-p for p in self.parameters.gradient(vmap)] # minus sign is important!
        
    
class VarianceTargetParamUpdater(ParamUpdater):
    def calculate_update(self):
        pass # TODO LATER
        
        
    
    
"""
DecayParamUpdater(Parameters p)
  * MomentumParamUpdater(ParamUpdater pu)
  * CDParamUpdater(Parameters p, StatsCollector s)
  * SparsityTargetParamUpdater(Parameters p, StatsCollector s, target) # this also needs stats!
  * VarianceTargetParamUpdater(Parameters p, target) # maybe, as an exercise, since it doesn't really work anyway
  * SumParamUpdater([ParamUpdater p])
  * ScaleParamUpdater([ParamUpdater p], scaling_factor)
"""

