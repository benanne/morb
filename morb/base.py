import theano
import theano.tensor as T

def _unique(l): # no idea why this function isn't available - the set() trick only works for hashable types!
    u = []
    for e in l:
        if e not in u:
            u.append(e)
    return u

### Base classes: sampling and modeling ###
    
class Units(object):    
    def __init__(self, rbm, name=None):
        self.rbm = rbm
        self.name = name
        self.proxy_units = [] # list of units that are proxies of this Units instance
        self.rbm.add_units(self)
        
    def activation(self, vmap):
        terms = [param.activation_term_for(self, vmap) for param in self.rbm.params_affecting(self)]
        # the linear activation is the sum of the activations for each of the parameters.
        return sum(terms, T.constant(0, theano.config.floatX))
        
    def sample_from_activation(self, vmap):
        raise NotImplementedError("Sampling not supported for this Units instance: %s" % repr(self))
        
    def mean_field_from_activation(self, vmap):
        raise NotImplementedError("Mean field not supported for this Units instance: %s" % repr(self))
        
    def free_energy_term_from_activation(self, vmap):
        raise NotImplementedError("Free energy calculation not supported for this Units instance: %s" % repr(self))
        
    def log_prob_from_activation(self, vmap, activation_vmap):
        raise NotImplementedError("Log-probability calculation not supported for this Units instance: %s" % repr(self))
        # note that this gives the log probability density for continuous units, but the log probability mass for discrete ones.
                
    def sample(self, vmap):
        return self.sample_from_activation({ self: self.activation(vmap) })
        
    def mean_field(self, vmap):
        return self.mean_field_from_activation({ self: self.activation(vmap) })
        
    def free_energy_term(self, vmap):
        return self.free_energy_term_from_activation({ self: self.activation(vmap) })
        
    def log_prob(self, vmap):
        activation_vmap = { self: self.activation(vmap) }
        return self.log_prob_from_activation(vmap, activation_vmap)
        
    def __repr__(self):
        return "<morb:Units '%s'>" % self.name


class ProxyUnits(Units):
    def __init__(self, rbm, units, func, name=None):
        super(ProxyUnits, self).__init__(rbm, name)
        self.units = units # the units this proxy is a function of
        self.func = func # the function to apply
        # simple proxy units do not support mean field, the class needs to be overridden for this.
        
    def sample(self, vmap):
        s = self.units.sample(vmap)
        return self.func(s)
        
    def sample_from_activation(self, vmap):
        s = self.units.sample_from_activation(vmap)
        return self.func(s)
        
    def mean_field(self, vmap):
        m = self.units.mean_field(vmap)
        return self.func(m)
        
    def mean_field_from_activation(self, vmap):
        m = self.units.mean_field_from_activation(vmap)
        return self.func(m)
        
        
class Parameters(object):
    def __init__(self, rbm, units_list, name=None):
        self.rbm = rbm
        self.units_list = units_list
        self.terms = {} # terms is a dict of FUNCTIONS that take a vmap.
        self.energy_gradients = {} # a dict of FUNCTIONS that take a vmap.
        self.energy_gradient_sums = {} # a dict of FUNCTIONS that take a vmap.
        self.name = name
        self.rbm.add_parameters(self)
        
    def activation_term_for(self, units, vmap):
        return self.terms[units](vmap)
        
    def energy_gradient_for(self, variable, vmap):
        """
        Returns the energy gradient for each example in the batch.
        """
        return self.energy_gradients[variable](vmap)
        
    def energy_gradient_sum_for(self, variable, vmap):
        """
        Returns the energy gradient, summed across the minibatch dimension.
        If a fast implementation for this is available in the energy_gradient_sums
        dictionary, this will be used. Else the energy gradient will be computed
        for each example in the batch (using the implementation from the
        energy_gradients dictionary) and then summed.
        
        Take a look at the ProdParameters implementation for an example of where
        this is useful: the gradient summed over the batch can be computed more
        efficiently with a dot product.
        """
        if variable in self.energy_gradient_sums:
            return self.energy_gradient_sums[variable](vmap)
        else:
            return T.sum(self.energy_gradients[variable](vmap), axis=0)
        
    def energy_term(self, vmap):
        raise NotImplementedError("Parameters base class")
        
    def affects(self, units):
        return (units in self.units_list)
        
    def __repr__(self):
        units_names = ", ".join(("'%s'" % u.name) for u in self.units_list)
        return "<morb:Parameters '%s' affecting %s>" % (self.name, units_names)


### Base classes: training (parameter updates) ###


class Updater(object):
    # An Updater object updates a single parameter variable. Multiple Updaters can compute updates for a single variable, which can then be aggregated by composite Updaters (like the SumUpdater)
    def __init__(self, variable, stats_list=[]):
        # variable is a single parameter variable, not a Parameters object or a list of variables.
        self.variable = variable
        self.stats_list = stats_list
        self.theano_updates = {} # some Updaters have state. Most don't, so then this is just
        # an empty dictionary. Those who do have state (like the MomentumUpdater) override
        # this variable.
                
    def get_update(self):
        raise NotImplementedError("Updater base class")
        
    def get_theano_updates(self):
        """
        gets own updates and the updates of all contained updaters (if applicable).
        """
        return self.theano_updates
        
    def _to_updater(self, e):
        """
        helper function that turns any expression into an updater
        """
        if not isinstance(e, Updater):
            eu = ExpressionUpdater(self.variable, e)
            return eu
        else:
            return e
        
    def __add__(self, p2):
        p2 = self._to_updater(p2)
        return SumUpdater([self, p2])
        
    def __sub__(self, p2):
        p2 = self._to_updater(p2)
        return self + (-p2)
        
    __radd__ = __add__
    __rsub__ = __sub__
        
    def __neg__(self):
        return ScaleUpdater(self, -1)
        
    def __mul__(self, a):
        return ScaleUpdater(self, a)
        # a is assumed to be a scalar!
        
    def __div__(self, a):
        return self * (1.0/a)
        
    __rmul__ = __mul__
    __rdiv__ = __div__
        

# this extension has to be here because it's used in the base class
class ExpressionUpdater(Updater):
    """
    An updater that returns a specified expression as its update.
    Mainly useful internally.
    """
    def __init__(self, variable, expression):
        super(ExpressionUpdater, self).__init__(variable)
        self.expression = expression
        
    def get_update(self):
        return self.expression

     
# this extension has to be here because it's used in the base class        
class ScaleUpdater(Updater):
    def __init__(self, pu, scaling_factor):
        super(ScaleUpdater, self).__init__(pu.variable, pu.stats_list)
        self.pu = pu
        self.scaling_factor = scaling_factor
        
    def get_update(self):
        return self.scaling_factor * self.pu.get_update()
        
    def get_theano_updates(self):
        u = {} # a scale updater has no state, so it has no theano updates of its own.
        u.update(self.pu.get_theano_updates())
        return u
        
# this extension has to be here because it's used in the base class
class SumUpdater(Updater):
    def __init__(self, updaters):
        # assert that all updaters affect the same variable, gather stats collectors
        self.updaters = updaters
        stats_list = []
        for pu in updaters:
            if pu.variable != updaters[0].variable:
                raise RuntimeError("Cannot add Updaters that affect a different variable together")        
            stats_list.extend(pu.stats_list)
        stats_list = _unique(stats_list) # we only need each Stats object once.
        
        super(SumUpdater, self).__init__(updaters[0].variable, stats_list)
        
    def get_update(self):
        return sum((pu.get_update() for pu in self.updaters), T.constant(0, theano.config.floatX))
        
    def get_theano_updates(self):
        u = {} # a sum updater has no state, so it has no theano updates of its own.
        for pu in self.updaters:
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

    def get_theano_updates(self, train=True):
        theano_updates = {}
        # collect stats
        stats_list = _unique([s for pu in self.umap.values() for s in pu.stats_list]) # cannot use set() here because dicts are not hashable.
        for s in stats_list:
            theano_updates.update(s.get_theano_updates())
        
        if train:
            variable_updates = {}
            for v, pu in self.umap.items():
                theano_updates.update(pu.get_theano_updates()) # Updater state updates
                theano_updates[v] = pu.get_update() # variable update
                
        return theano_updates
                

### Base classes: RBM container class ###

class RBM(object):
    def __init__(self):
        self.units_list = []
        self.params_list = []
        
    def add_units(self, units):
        self.units_list.append(units)
        
    def remove_units(self, units):
        self.units_list.remove(units)
        
    def add_parameters(self, params):
        self.params_list.append(params)
        
    def remove_parameters(self, params):
        self.params_list.remove(params)
        
    @property
    def variables(self):
        """
        property that returns a set of all parameter variables.
        """
        # This is a set, because if it were a regular list,
        #there would be duplicates when parameters are tied.
        return set(variable for params in self.params_list for variable in params.variables)
                
    def params_affecting(self, units):
        """
        return a list of all Parameters that contribute a term to the activation of Units units.
        """
        return [param for param in self.params_list if param.affects(units)]
        
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
        
    def energy_gradient(self, variable, vmap):
        """
        sums the gradient contributions of all Parameters instances for the given variable.
        """
        return sum((p.energy_gradient_for(variable, vmap) for p in self.params_list if variable in p.variables), T.constant(0, theano.config.floatX))
        
    def energy_gradient_sum(self, variable, vmap):
        """
        sums the gradient contributions of all Parameters instances for the given variable,
        where the contributions are summed over the minibatch dimension.
        """
        return sum((p.energy_gradient_sum_for(variable, vmap) for p in self.params_list if variable in p.variables), T.constant(0, theano.config.floatX))
    
    def energy_terms(self, vmap):
        return [params.energy_term(vmap) for params in self.params_list]
        
    def energy(self, vmap):
        # the energy is the sum of the energy terms for each of the parameters.
        return sum(self.energy_terms(vmap), T.constant(0, theano.config.floatX))
        
    def complete_units_list_split(self, units_list):
        """
        Returns two lists: one with basic units and one with proxy units.
        For all basic units in the units_list, all proxies are added as well.
        For all proxy units in the units_list, all missing basic units are added
        as well.
        """
        proxy_units, basic_units = [], []        
        for u in units_list:
            if isinstance(u, ProxyUnits):
                if u not in proxy_units:
                    proxy_units.append(u)
                if u.units not in basic_units:
                    basic_units.append(u.units)
            else:
                if u not in basic_units:
                    basic_units.append(u)
                for p in u.proxy_units:
                    if p not in proxy_units:
                        proxy_units.append(p)
                        
        return basic_units, proxy_units
        
    def complete_units_list(self, units_list):
        b, pr = self.complete_units_list_split(units_list)
        return b + pr
        
    def complete_vmap(self, vmap):
        """
        Takes in a vmap and computes any missing proxy units values.
        """
        vmap = vmap.copy() # don't modify the original dict
        units_list = vmap.keys()
        missing_units_list = []
        for u in units_list:
            for p in u.proxy_units:
                if p not in units_list:
                    missing_units_list.append(p)
                    
        for p in missing_units_list:
            vmap[p] = p.func(vmap[p.units])
            
        return vmap
    
    def sample_from_activation(self, vmap):
        """
        This method allows to sample a given set of Units instances at the same time and enforces consistency.
        say v is a Units instance, and x is a ProxyUnits instance tied to v. Then the following:
        vs = v.sample_from_activation(a1)
        xs = x.sample_from_activation(a2)
        ...will yield inconsistent samples (i.e. xs != func(vs)). This is undesirable in CD, for example.
        To remedy this, only the 'basic' units are sampled, and the values of the proxy units are computed.
        The supplied activation_map is assumed to be complete.
        """
        # This code does not support proxies of proxies.
        # If this should be supported, this code should be rethought.
        
        # first, 'complete' units_list: if there are any proxies whose basic units
        # are not included, add them. If any of the included basic units have proxies
        # which are not, add them as well. Make two separate lists.
        # note that this completion comes at almost no extra cost: the expressions
        # are added to the resulting dictionary, but that doesn't mean they'll
        # necessarily be used (and thus, compiled).
        units_list = vmap.keys()
        basic_units, proxy_units = self.complete_units_list_split(units_list)
                
        # sample all basic units
        samples = {}
        for u in basic_units:
            samples[u] = u.sample_from_activation(vmap)
            
        # compute all proxy units
        for u in proxy_units:
            samples[u] = u.func(samples[u.units])
        
        return samples
    
    def sample(self, units_list, vmap):
        """
        This method allows to sample a given set of Units instances at the same time and enforces consistency.
        say v is a Units instance, and x is a ProxyUnits instance tied to v. Then the following:
        vs = v.sample(vmap)
        xs = x.sample(vmap)
        ...will yield inconsistent samples (i.e. xs != func(vs)). This is undesirable in CD, for example.
        To remedy this, only the 'basic' units are sampled, and the values of the proxy units are computed.
        All proxies are always included in the returned vmap.
        """
        activations_vmap = self.activations(units_list, vmap)
        return self.sample_from_activation(activations_vmap)
    
    def mean_field_from_activation(self, vmap):
        units_list = vmap.keys()
        units_list = self.complete_units_list(units_list)
        # no consistency need be enforced when using mean field.
        return dict((u, u.mean_field_from_activation(vmap)) for u in units_list)
    
    def mean_field(self, units_list, vmap):
        activations_vmap = self.activations(units_list, vmap)
        return self.mean_field_from_activation(activations_vmap)
        
    def free_energy_unchanged_terms(self, units_list, vmap):
        """
        The terms of the energy that don't involve any of the given units.
        These terms are unchanged when computing the free energy, where
        the given units are integrated out.
        """
        unchanged_terms = []
        for params in self.params_list:
            if not any(params.affects(u) for u in units_list):
                # if none of the given Units instances are affected by the current Parameters instance,
                # this term is unchanged (it's the same as in the energy function)
                unchanged_terms.append(params.energy_term(vmap))
        
        return unchanged_terms
        
    def free_energy_affected_terms_from_activation(self, vmap):
        """
        For each Units instance in the activation vmap, the corresponding free energy
        term is returned.
        """
        return dict((u, u.free_energy_term_from_activation(vmap)) for u in vmap)

    def free_energy_affected_terms(self, units_list, vmap):
        """
        The terms of the energy that involve the units given in units_list are
        of course affected when these units are integrated out. This method
        gives the 'integrated' terms.
        """
        return dict((u, u.free_energy_term(vmap)) for u in units_list)
            
    def free_energy(self, units_list, vmap):
        """
        Calculates the free energy, integrating out the units given in units_list.
        This has to be a list of Units instances that are independent of eachother
        given the other units, and each of them has to have a free_energy_term.
        """
        # first, get the terms of the energy that don't involve any of the given units. These terms are unchanged.
        unchanged_terms = self.free_energy_unchanged_terms(units_list, vmap)
        # all other terms are affected by the summing out of the units.        
        affected_terms = self.free_energy_affected_terms(units_list, vmap).values()  
        # note that this separation breaks down if there are dependencies between the Units instances given.
        return sum(unchanged_terms + affected_terms, T.constant(0, theano.config.floatX))
        
    def activations(self, units_list, vmap):
        units_list = self.complete_units_list(units_list)
        # no consistency need be enforced when computing activations.
        return dict((u, u.activation(vmap)) for u in units_list)

    def __repr__(self):
        units_names = ", ".join(("'%s'" % u.name) for u in self.units_list)
        params_names = ", ".join(("'%s'" % p.name) for p in self.params_list)
        return "<morb:%s with units %s and parameters %s>" % (self.__class__.__name__, units_names, params_names)
        

