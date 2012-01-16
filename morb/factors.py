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
class ThirdOrderFactoredParameters(Parameters):
    \"\"\"
    Factored 3rd order parameters, connecting three Units instances. Each factored
    parameter matrix has dimensions (units_size, num_factors).
    \"\"\"
    def __init__(self, rbm, units_list, variables, name=None):
        super(ThirdOrderFactoredParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 3
        assert len(variables) == 3
        self.variables = variables
        self.var0 = variables[0]
        self.var1 = variables[1]
        self.var2 = variables[2]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]
        self.prod0 = lambda vmap: T.dot(vmap[self.u0], self.var0) # (mb, f)
        self.prod1 = lambda vmap: T.dot(vmap[self.u1], self.var1) # (mb, f)
        self.prod2 = lambda vmap: T.dot(vmap[self.u2], self.var2) # (mb, f)
        self.terms[self.u0] = lambda vmap: T.dot(self.prod1(vmap) * self.prod2(vmap), self.var0.T) # (mb, u0)
        self.terms[self.u1] = lambda vmap: T.dot(self.prod0(vmap) * self.prod2(vmap), self.var1.T) # (mb, u1)
        self.terms[self.u2] = lambda vmap: T.dot(self.prod0(vmap) * self.prod1(vmap), self.var2.T) # (mb, u2)
                
        self.energy_gradients[self.var0] = lambda vmap: T.dot(vmap[self.u0].T, self.prod1(vmap) * self.prod2(vmap)) # (u0, f)
        self.energy_gradients[self.var1] = lambda vmap: T.dot(vmap[self.u1].T, self.prod0(vmap) * self.prod2(vmap)) # (u1, f)
        self.energy_gradients[self.var2] = lambda vmap: T.dot(vmap[self.u2].T, self.prod0(vmap) * self.prod1(vmap)) # (u2, f)
        # the T.dot also sums out the minibatch dimension
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.u1](vmap) * vmap[self.u1])
        # sum is over the minibatch and the u1 dimension.
"""
