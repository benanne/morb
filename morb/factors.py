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

# TODO: an uninitialised factor typically just results in bogus results, it doesn't
# raise any exceptions. This isn't very clean. Maybe find a way around this.

class Factor(Parameters):
    """
    A 'factor' can be used to construct factored parameters from other
    Parameters instances.
    """
    def __init__(self, rbm, name=None):
        super(Factor, self).__init__(rbm, [], name=name)
        # units_list is initially empty, but is expanded later by adding Parameters.
        self.variables = [] # same for variables
        self.params_list = []
        self.terms = {}
        self.energy_gradients = {} # careful, this is now a dict of LISTS to support parameter tying.
        self.energy_gradient_sums = {} # same here!
        self.initialized = False
        
    def check_initialized(self):
        if not self.initialized:
            raise RuntimeError("Factor '%s' has not been initialized." % self.name)
    
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
    
    def update_terms(self, params):
        """
        Add activation terms for the units associated with Parameters instance params
        """
        ul = list(params.units_list) # copy
        ul.remove(self)
        for u in ul:
            def term(vmap):
                fp = self.factor_product(params, vmap) # compute factor values
                fvmap = vmap.copy()
                fvmap.update({ self: fp }) # insert them in a vmap copy
                return params.terms[u](fvmap) # pass the copy to the Parameters instance so it can compute its activation
            self.terms[u] = term

    def update_energy_gradients(self, params):
        """
        Add/update energy gradients for the variables associated with Parameters instance params
        """
        for var in params.variables:
            def grad(vmap):
                fp = self.factor_product(params, vmap) # compute factor values
                fvmap = vmap.copy()
                fvmap.update({ self: fp }) # insert them in a vmap copy
                return params.energy_gradient_for(var, fvmap)
                
            def grad_sum(vmap):
                fp = self.factor_product(params, vmap) # compute factor values
                fvmap = vmap.copy()
                fvmap.update({ self: fp }) # insert them in a vmap copy
                return params.energy_gradient_sum_for(var, fvmap)
           
            if var not in self.energy_gradients:
                self.energy_gradients[var] = []
                self.energy_gradient_sums[var] = []
            self.energy_gradients[var].append(grad)
            self.energy_gradient_sums[var].append(grad_sum)

    def activation_term_for(self, units, vmap):
        self.check_initialized()
        return self.terms[units](vmap)
        
    def energy_gradient_for(self, variable, vmap):
        self.check_initialized()
        return sum(f(vmap) for f in self.energy_gradients[variable]) # sum all contributions
        
    def energy_gradient_sum_for(self, variable, vmap):
        self.check_initialized()
        return sum(f(vmap) for f in self.energy_gradient_sums[variable]) # sum all contributions

    def energy_term(self, vmap):
        """
        The energy term of the factor, which is the product of all activation
        terms of the factor from the contained Parameters instances.
        """
        self.check_initialized()
        factor_activations = [params.terms[self](vmap) for params in self.params_list]
        return T.sum(reduce(mul, factor_activations))
    
    def initialize(self):
        """
        Extract Units instances and variables from each contained Parameters
        instance. Unfortunately there is no easy way to do this automatically
        when the Parameters instances are created, because the add_parameters
        method is called before they are fully initialised.
        """
        if self.initialized: # don't initialize multiple times
            return # TODO: maybe this should raise a warning?
            
        for params in self.params_list:
            self.variables.extend(params.variables)
            units_list = list(params.units_list)
            units_list.remove(self)
            self.units_list.extend(units_list)
            self.update_terms(params)
            self.update_energy_gradients(params)
            
        self.initialized = True
        
    def add_parameters(self, params):
        """
        This method is called by the Parameters constructor when the 'rbm'
        argument is substituted for a Factor instance.
        """
        self.params_list.append(params)

    def __repr__(self):
        units_names = ", ".join(("'%s'" % u.name) for u in self.units_list)
        return "<morb:Factor '%s' affecting %s>" % (self.name, units_names)

