import theano.tensor as T
from morb.base import Parameters


# TODO: general 'Factor' implementation that can represent factored parameters by combining other types of 2D parameters. This could be used to implement a factored convolutional RBM or something, or an n-way factored RBM with n >= 2.

# The idea is that the 'Factor' object acts as an RBM and Units proxy for the contained Parameters, and is used as a Parameters object within the RBM.

# The Parameters contained within the Factor MUST NOT be added to the RBM, because their joint energy term is not linear in each of the individual factored parameter sets (but rather multiplicative). Adding them to the RBM would cause them to contribute an energy term, which doesn't make sense.

class Factor(Parameters):
    """
    A 'factor' can be used to construct factored parameters from other Parameters.
    """
    def __init__(self, rbm, name=None):
        super(Factor, self).__init__(rbm, [], name=name)
        # units_list is initially empty, but is expanded later by adding Parameters.
        self.variables = [] # same for variables
        self.params_list = []
        # TODO: define the terms
        # TODO: define the energy term
        # TODO: define the energy gradients of the respective parameters... hmm.
        
    def add_parameters(self, params):
        """
        This method is called by the Parameters constructor when the 'rbm'
        argument is substituted for a Factor instance.
        """
        self.params_list.append(params)
        # TODO: update 'variables' and 'units_list' here?
    
    
    
    
"""
class ThirdOrderParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ThirdOrderParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 3
        self.var = W
        self.variables = [self.var]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]
        
        def term_u0(vmap):
            p = tensordot(vmap[self.u1], W, axes=([1],[1])) # (mb, u0, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u0)
            # cannot use two tensordots here because of the minibatch dimension.
            
        def term_u1(vmap):
            p = tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u1)
            
        def term_u2(vmap):
            p = tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u1].dimshuffle(0, 1, 'x'), axis=1) # (mb, u2)
            
        self.terms[self.u0] = term_u0
        self.terms[self.u1] = term_u1
        self.terms[self.u2] = term_u2
                
        def gradient(vmap):
            p = vmap[self.u0].dimshuffle(0, 1, 'x') * vmap[self.u1].dimshuffle(0, 'x', 1) # (mb, u0, u1)
            p2 = p.dimshuffle(0, 1, 2, 'x') * vmap[self.u2].dimshuffle(0, 'x', 'x', 1) # (mb, u0, u1, u2)
            return T.sum(p2, axis=0) # sum out minibatch dimension
            
        self.energy_gradients[self.var] = gradient
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.u1](vmap) * vmap[self.u1])
        # sum is over the minibatch and the u1 dimension.
"""
