import theano
import theano.tensor as T

def _unique(l): # no idea why this function isn't available - the set() trick only works for hashable types!
    u = []
    for e in l:
        if e not in u:
            u.append(e)
    return u

### Base classes: sampling and modeling ###

class ActivationFunction(object):
    def apply(self, x):
        raise NotImplementedError("ActivationFunction base class")
    
class Sampler(object):
    def apply(self, a):
        raise NotImplementedError("Sampler base class")
    
class Units(object):
    # container class for a theano variable representing unit values, an ActivationFunction and a Sampler.
    # a unit could just as well be represented as a tuple (u, ActivationFunction(), Sampler(), [CDSampler()]) but it's probably
    # nicer to have a container class for this.
    
    def __init__(self, rbm, activation_function, sampler, name=None):
        self.rbm = rbm
        self.activation_function = activation_function
        self.sampler = sampler
        self.name = name
        self.rbm.add_units(self)
    
    def linear_activation(self, vmap):
        # args and kwargs represent the other Units of the RBM that these units are tied to.
        terms = [param.activation_term_for(self, vmap) for param in self.rbm.params_affecting(self)]
        # the linear activation is the sum of the activations for each of the parameters.
        return sum(terms)
        
    def activation(self, vmap):
        l = self.linear_activation(vmap)
        return self.activation_function.apply(l)
            
    def sample(self, vmap, **kwargs):
        a = self.activation(vmap)
        return self.sampler.apply(a, **kwargs)
        
    def free_energy_term(self, vmap):
        raise NotImplementedError("Free energy calculation not supported for this Units instance")
        
    def __repr__(self):
        return "<morb:Units '%s' with sampler '%s' and activation function '%s'>" % (self.name, self.sampler.name, self.activation_function.name)
          

# decorators to create sampler and activation function objects from functions
def sampler(func):
    class s(Sampler):
        def apply(self, a, *args, **kwargs):
            return func(a, *args, **kwargs)
    inst = s()
    inst.name = func.__name__
    return inst
    
def activation_function(func):
    class f(ActivationFunction): 
        def apply(self, x, *args, **kwargs):
            return func(x, *args, **kwargs)
    inst = f()
    inst.name = func.__name__
    return inst

# function to quickly create a basic units class
def units_type(activation_function, sampler):
    class U(Units):
        def __init__(self, rbm, name=None):
            super(U, self).__init__(rbm, activation_function, sampler, name=name)
            
    return U
        
class Parameters(object):
    def __init__(self, rbm, units_list, name=None):
        self.rbm = rbm
        self.units_list = units_list
        self.terms = {} # terms is a dict of FUNCTIONS that take a vmap variable mapping dict.
        self.name = name
        self.rbm.add_parameters(self)
        
    def activation_term_for(self, units, vmap):
        return self.terms[units](vmap)
        
    def energy_term(self, vmap):
        raise NotImplementedError("Parameters base class")
        
    def gradient(self, vmap):
        # gives the parameter update in terms of the statistics
        raise NotImplementedError("Parameters base class")
        
    # @property
    # def variables(self):
    #     raise NotImplementedError("Parameters base class")
    # # cannot do this because then it can't be assigned in subclasses :(
        
    def affects(self, units):
        return (units in self.units_list)
        
    def affects_all(self, units_list):
        return all(self.affects(units) for units in units_list)
        
    def __repr__(self):
        units_names = ", ".join(("'%s'" % u.name) for u in self.units_list)
        return "<morb:Parameters '%s' affecting %s>" % (self.name, units_names)


### Base classes: training (parameter updates) ###


class ParamUpdater(object):
    # A ParamUpdater object updates a single Parameters object. Multiple ParamUpdaters can compute updates for a single Parameters object, which can then be aggregated by composite ParamUpdaters (like the SumParamUpdater)
    def __init__(self, parameters, stats_list=[]):
        # parameters is A SINGLE Parameters object. not a list.
        self.parameters = parameters
        self.stats_list = stats_list
        self.theano_updates = {} # some ParamUpdaters have state. Most don't, so then this is just
        # an empty dictionary. Those who do have state (like the MomentumParamUpdater) override
        # this variable.
                
    def get_update(self):
        raise NotImplementedError("ParamUpdater base class")
        
    def get_theano_updates(self):
        """
        gets own updates and the updates of all contained paramupdaters (if applicable).
        """
        return self.theano_updates
        
    def __add__(self, p2):
        return SumParamUpdater([self, p2])
        
    def __neg__(self):
        return ScaleParamUpdater(self, -1)
        
    def __sub__(self, p2):
        return self + (-p2)
        
    def __rmul__(self, a):
        return ScaleParamUpdater(self, a)
        # a is assumed to be a scalar!
        
    def __mul__(self, a):
        return ScaleParamUpdater(self, a)
        # a is assumed to be a scalar!
        
    def __div__(self, a):
        return self * (1.0/a)
        
    def __rdiv__(self, a):
        return self * (1.0/a)
        

# this extension has to be here because it's used in the base class        
class ScaleParamUpdater(ParamUpdater):
    def __init__(self, pu, scaling_factor):
        super(ScaleParamUpdater, self).__init__(pu.parameters, pu.stats_list)
        self.pu = pu
        self.scaling_factor = scaling_factor
        
    def get_update(self):
        return [self.scaling_factor * d for d in self.pu.get_update()]
        
    def get_theano_updates(self):
        u = {} # a scale param_updater has no state, so it has no theano updates of its own.
        u.update(self.pu.get_theano_updates())
        return u

# this extension has to be here because it's used in the base class
class SumParamUpdater(ParamUpdater):
    def __init__(self, param_updaters):
        # assert that all param_updaters affect the same Parameters object, gather stats collectors
        self.param_updaters = param_updaters
        stats_list = []
        for pu in param_updaters:
            if pu.parameters != param_updaters[0].parameters:
                raise RuntimeError("Cannot add ParamUpdaters that affect a different Parameters object together")        
            stats_list.extend(pu.stats_list)
        stats_list = _unique(stats_list) # we only need each Stats object once.
        
        super(SumParamUpdater, self).__init__(param_updaters[0].parameters, stats_list)
        
    def get_update(self):
        updaters = self.param_updaters[:] # make a copy
        first_updater = updaters.pop(0)
        s = first_updater.get_update()
        for pu in updaters: # iterate over the REMAINING updaters
            s = [sd + d for sd, d in zip(s, pu.get_update())]
        return s
        
    def get_theano_updates(self):
        u = {} # a sum param_updater has no state, so it has no theano updates of its own.
        for pu in self.param_updaters:
            u.update(pu.get_theano_updates())
        return u
        

class Stats(dict): # a stats object is just a dictionary of vmaps, but it also holds associated theano updates.
    def __init__(self, updates):
        self.theano_updates = updates
    
    def get_theano_updates(self):
        return self.theano_updates

class Trainer(object):
    def __init__(self, rbm, umap):
        self.rbm = rbm
        self.umap = umap

    def get_theano_updates(self, vmap, train=True):
        theano_updates = {}
        # collect stats
        stats_list = _unique([s for pu in self.umap.values() for s in pu.stats_list]) # cannot use set() here because dicts are not hashable.
        for s in stats_list:
            theano_updates.update(s.get_theano_updates())
        
        if train:
            # calculate updates
            for p, pu in self.umap.items():
                updated_variables = [v + u for v, u in zip(p.variables, pu.get_update())]
                theano_updates.update(dict(zip(p.variables, updated_variables))) # variable updates
                theano_updates.update(pu.get_theano_updates()) # ParamUpdater state updates
                # (MomentumParamUpdater has state, for example)

        return theano_updates


### Base classes: RBM container class ###

class RBM(object):
    def __init__(self):
        self.units_list = []
        self.params_list = []
        
    def add_units(self, units):
        self.units_list.append(units)
        
    def add_parameters(self, params):
        self.params_list.append(params)
                
    def params_affecting(self, units):
        """
        return a list of all Parameters that contribute a term to the activation of Units units.
        """
        return [param for param in self.params_list if param.affects(units)]
        
    def params_affecting_all(self, units_list):
        """
        return a list of all Parameters that contribute a term to the activations of ALL Units in the given units_list.
        """
        return [param for param in self.params_list if param.affects_all(units_list)]
        
    def dependent_units(self, given_units_list):
        """
        returns a list of all Units that are dependent on the given list of Units.
        this is useful for block Gibbs sampling (where given_units_list is the set of
        visible Units, and the set of hiddens is returned).
        
        Note that this method does not detect possible dependencies between the given Units
        themselves, or the returned Units themselves! This check should be performed before
        doing gibbs sampling. Alternatively, some back and forth sampling between these
        dependent Units can be used (what's the correct name for this again?)
        
        Also, context (i.e. units that should never be sampled) has to be handled separately.
        This method will include context as it only checks which units are linked to which
        other units, and these links are not directional.
        """
        # first, find all the parameters affecting the Units in the given_units_list
        # then, for each of these, add all the affected units
        dependent_units_list = []
        for u in given_units_list:
            params_list = self.params_affecting(u)
            for params in params_list:
                dependent_units_list.extend(params.units_list)
                
        # finally, remove the given units and return the others
        return set([u for u in dependent_units_list if u not in given_units_list])
        # note that there are no dependency checks here.
        
    def energy(self, vmap):
        terms = [params.energy_term(vmap) for params in self.params_list]
        # the energy is the sum of the energy terms for each of the parameters.
        return sum(terms)
        
    def free_energy(self, units_list, vmap):
        """
        Calculates the free energy with respect to the units given in units_list.
        This has to be a list of Units instances that are independent of eachother
        given the other units, and each of them has to have a free_energy_term.
        """
        
        # first, get the terms of the energy that don't involve any of the given units. These terms are unchanged.
        unchanged_terms = []
        for params in self.params_list:
            if not any(params.affects(u) for u in units_list):
                # if none of the given Units instances are affected by the current Parameters instance,
                # this term is unchanged (it's the same as in the energy function)
                unchanged_terms.append(params.energy_term(vmap))

        # all other terms are affected by the summing out of the units.        
        affected_terms = [u.free_energy_term(vmap) for u in units_list]
        
        # note that this separation breaks down if there are dependencies between the Units instances given.
        return sum(unchanged_terms + affected_terms)

    def __repr__(self):
        units_names = ", ".join(("'%s'" % u.name) for u in self.units_list)
        params_names = ", ".join(("'%s'" % p.name) for p in self.params_list)
        return "<morb:%s with units %s and parameters %s>" % (self.__class__.__name__, units_names, params_names)
        

