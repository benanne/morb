from morb import base
import theano.tensor as T

# Tools for sampling RBMs at different temperatures

class TemperedUnits(base.Units):
    """
    Encapsulates another Units instance and allows for samples to be drawn at
    different temperatures. A temperature vector should be specified at
    instantiation. Note that this can be a Theano shared variable, so it is
    possible to change temperatures afterwards.
    
    Important: creating a TemperedUnits instance removes the original instance
    from the RBM and replaces it.
    
    When sampling from the units, the activations will be rescaled according
    to the given temperature vector. The temperatures are mapped across the
    first dimension (minibatch axis). This means that the number of activations
    supplied should equal the size of this dimension.
    """
    def __init__(self, units, temperatures):
        self.units = units
        self.temperatures = temperatures
        self.units.rbm.remove_units(self.units) # remove encapsulated units from RBM
        # They will be replaced by this instance.
        super(TemperedUnits, self).__init__(self.units.rbm, name=self.units.name)
    
    def tempered_activation(self, vmap):
        a = self.units.activation(vmap)
        return T.shape_padright(self.temperatures, a.ndim - 1) * a # multiply along minibatch axis
        
    def sample(self, vmap):
        return self.sample_from_activation(self.tempered_activation(vmap))
        
    def mean_field(self, vmap):
        return self.mean_field_from_activation(self.tempered_activation(vmap))
        
    def free_energy_term(self, vmap):
        return self.free_energy_term_from_activation(self.tempered_activation(vmap))
