from morb.base import Monitor

import theano
import theano.tensor as T


class ReconstructionMSEMonitor(Monitor):
    def __init__(self, stats_collector, u):
        self.stats_collector = stats_collector
        self.u = u # the Units instance for which the reconstruction error should be computed.
        
    def expression(self):
        data = self.stats_collector.stats['data'][self.u]
        reconstruction = self.stats_collector.stats['model'][self.u]
        return T.mean((data - reconstruction) ** 2)
    
    
class ReconstructionCrossEntropyMonitor(Monitor):
    def __init__(self, stats_collector, u):
        self.stats_collector = stats_collector
        self.u = u
        
    def expression(self):
        data = self.stats_collector.stats['data'][self.u]
        reconstruction_linear = self.stats_collector.stats['model_linear_activation'][self.u]
        return T.mean(T.sum(data*T.log(T.nnet.sigmoid(reconstruction_linear)) +
                      (1 - data)*T.log(1 - T.nnet.sigmoid(reconstruction_linear)), axis=1))
                      
        # without optimisation:
        # return T.mean(T.sum(data*T.log(reconstruction) + (1 - data)*T.log(reconstruction), axis=1))
    
# 'trivial' monitors
class ReconstructionMonitor(Monitor):
    def __init__(self, stats_collector, u):
        self. stats_collector = stats_collector
        self.u = u
        
    def expression(self):
        return self.stats_collector.stats['model'][self.u]

class DataMonitor(Monitor):
    def __init__(self, stats_collector, u):
        self. stats_collector = stats_collector
        self.u = u
        
    def expression(self):
        return self.stats_collector.stats['data'][self.u]
        
        
class DataEnergyMonitor(Monitor):
    def __init__(self, stats_collector, rbm):
        self. stats_collector = stats_collector
        self.rbm = rbm
        
    def expression(self):
        return self.rbm.energy(self.stats_collector.stats['data'])

class ReconstructionEnergyMonitor(Monitor):
    def __init__(self, stats_collector, rbm):
        self. stats_collector = stats_collector
        self.rbm = rbm
        
    def expression(self):
        return self.rbm.energy(self.stats_collector.stats['model'])

# TODO: pseudo likelihood? is that feasible?


