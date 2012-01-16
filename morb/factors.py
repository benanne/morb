import theano.tensor as T
from morb.base import Parameters

from operator import mul

# general 'Factor' implementation that can represent factored parameters by
# combining other types of parameters. This could be used to implement a
# factored convolutional RBM or something, or an n-way factored RBM with n >= 2.

# The idea is that the 'Factor' object acts as an RBM and Units proxy for the
# contained Parameters, and is used as a Parameters object within the RBM.

# The Parameters contained within the Factor MUST NOT be added to the RBM,
# because their joint energy term is not linear in each of the individual
# factored parameter sets (but rather multiplicative). Adding them to the
# RBM would cause them to contribute an energy term, which doesn't make sense.

class Factor(Parameters):
    """
    A 'factor' can be used to construct factored parameters from other Parameters.
    """
    def __init__(self, rbm, name=None):
        super(Factor, self).__init__(rbm, [], name=name)
        # units_list is initially empty, but is expanded later by adding Parameters.
        self.variables = [] # same for variables
        self.params_list = []
        self.terms = {}
        self.units_map = {}
        self.energy_gradients = {}
    
    def factor_product_for(self, units, vmap):
        """
        The factor product needed to compute the activation of the given Units instance
        """
        params = self.units_map[units]
        return self.factor_product(params, vmap)
    
    def factor_product(self, params, vmap):
        """
        The factor product needed to compute the activation of the other units
        tied by Parameters params.
        """
        # get all Parameters except for the given instance
        fp_params_list = list(self.params_list) # make a copy
        fp_params_list.remove(params) # remove the given instance
        
        # compute activation terms of the factor
        activations = [fp_params.terms[self](vmap) for fp_params in fp_params_list]

        # multiply the activation terms
        return reduce(mul, activations)
    
    def update_units_map(self, params):
        """
        update the dict that maps units to the parameters that tie them to the factor,
        for the given Parameters instance params.
        """
        # get the params' units_list, remove the factor itself,
        # raise an error if the factor does not occur in the units_list.
        if self not in params.units_list:
            raise RuntimeError("Tried to update Factor units map with a Parameters instance that is not tied to the factor.")
        ul = list(params.units_list) # copy the list
        ul.remove(self) # get rid of the Factor itself.
        for u in ul:
            self.units_map[u] = params # update the mapping for each tied Units instance.
    
    def update_terms(self, params):
        # add activation terms for the units associated with Parameters instance params
        ul = list(params.units_list)
        ul.remove(self)
        for u in ul:
            def term(vmap):
                fp = self.factor_product(params, vmap) # compute factor values
                fvmap = vmap.copy()
                fvmap.update({ self: fp }) # insert them in a vmap copy
                return params.terms[u](fvmap) # pass the copy to the Parameters instance so it can compute its activation
            self.terms[u] = term

    def update_energy_gradients(self, params):
        # add energy gradients for the variables associated with Parameters instance params
        for var in params.variables:
            def grad(vmap):
                fp = self.factor_product(params, vmap) # compute factor values
                fvmap = vmap.copy()
                fvmap.update({ self: fp }) # insert them in a vmap copy
                return params.energy_gradients[var](fvmap)
            self.energy_gradients[var] = grad
    
    def energy_term(self, vmap):
        """
        The energy term of the factor, which is the product of all activation
        terms of the factor from the contained Parameters instances.
        """
        factor_activations = [params.terms[self](vmap) for params in self.params_list]
        return T.sum(reduce(mul, factor_activations))
       
    def add_parameters(self, params):
        """
        This method is called by the Parameters constructor when the 'rbm'
        argument is substituted for a Factor instance.
        """
        self.params_list.append(params)
        self.variables.extend(params.variables)
        self.units_list.extend(params.units_list)
        self.update_units_map(params)
        self.update_terms(params)
        self.update_energy_gradients(params)
    
    
# 'products' uit ThirdOrderFactoredParameters zijn eigenlijk de activaties van de 'factor' voor elk van de inkomende Parameters.
# # subparam.terms[self]
# hoe ziet een 'term' van de factor er dan uit? Er moet een term zijn voor elke units instantie!
# term = functino of params and f, linear in f.
# activation_term berekenen:
# - eerst de activaties van de factor berekenen, afkomstigvan de ANDERE parameterinstanties
# - dan deze als de 'waarden' van de factor beschouwen (in een vmap stoppen) en de activatie van de units in zijn respectieve parameters berekenen.
# er is een datastructuur nodig die units op de respectieve parameters mapt. Hier kan dan ook gedetecteerd worden of er ergens units dubbel gebruikt worden; maar als't context units zijn is dat eigenlijk geen probleem, dus dat moet enkel gedetecteerd kunnen worden, geen warnings of errors nodig.
    
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
