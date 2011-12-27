from morb.base import Stats

import numpy as np

import theano


def gibbs_step(rbm, vmap, units_list, mean_field_for_stats=[], mean_field_for_gibbs=[]):
    # implements a single gibbs step, and makes sure mean field is only used where it should be.
    # returns two vmaps, one for stats and one for gibbs.
    # also enforces consistency between samples, between the gibbs vmap and the stats vmap.
    # the provided lists and vmaps are expected to be COMPLETE. Otherwise, the behaviour is unspecified.
    
    # first, find out which units we need to sample for the stats vmap, and which for the gibbs vmap.
    # Mean field will be used for the others.
    units_sample_stats = units_list[:] # make a copy
    units_mean_field_stats = []
    for u in mean_field_for_stats:
        if u in units_sample_stats:
            units_sample_stats.remove(u) # remove all mean field units from the sample list
            units_mean_field_stats.append(u) # add them to the mean field list instead

    units_sample_gibbs = units_list[:]
    units_mean_field_gibbs = []
    for u in mean_field_for_gibbs:
        if u in units_sample_gibbs:
            units_sample_gibbs.remove(u) # remove all mean field units from the sample list
            units_mean_field_gibbs.append(u) # add them to the mean field list instead

    # now we can compute the total list of units to sample.
    # By sampling them all in one go, we can enforce consistency.
    units_sample = list(set(units_sample_gibbs + units_sample_stats))
    sample_vmap = rbm.sample(units_sample, vmap)
    units_mean_field = list(set(units_mean_field_gibbs + units_mean_field_stats))
    mean_field_vmap = rbm.mean_field(units_mean_field, vmap)
    
    # now, construct the gibbs and stats vmaps
    stats_vmap = dict((u, sample_vmap[u]) for u in units_sample_stats)
    stats_vmap.update(dict((u, mean_field_vmap[u]) for u in units_mean_field_stats))
    gibbs_vmap = dict((u, sample_vmap[u]) for u in units_sample_gibbs)
    gibbs_vmap.update(dict((u, mean_field_vmap[u]) for u in units_mean_field_gibbs))
        
    return stats_vmap, gibbs_vmap


def cd_stats(rbm, v0_vmap, visible_units, hidden_units, context_units=[], k=1, mean_field_for_stats=[], mean_field_for_gibbs=[], persistent_vmap=None):
    # mean_field_for_gibbs is a list of units for which 'mean_field' should be used during gibbs sampling, rather than 'sample'.
    # mean_field_for_stats is a list of units for which 'mean_field' should be used to compute statistics, rather than 'sample'.

    # complete units lists
    visible_units = rbm.complete_units_list(visible_units)
    hidden_units = rbm.complete_units_list(hidden_units)
    context_units = rbm.complete_units_list(context_units)
    
    # complete the supplied vmap
    v0_vmap = rbm.complete_vmap(v0_vmap)
    
    # extract the context vmap, because we will need to merge it into all other vmaps
    context_vmap = dict((u, v0_vmap[u]) for u in context_units)

    h0_activation_vmap = dict((h, h.activation(v0_vmap)) for h in hidden_units)
    h0_stats_vmap, h0_gibbs_vmap = gibbs_step(rbm, v0_vmap, hidden_units, mean_field_for_stats, mean_field_for_gibbs)
            
    # add context
    h0_activation_vmap.update(context_vmap)
    h0_gibbs_vmap.update(context_vmap)
    h0_stats_vmap.update(context_vmap)
    
    exp_input = [v0_vmap[u] for u in visible_units]
    exp_context = [v0_vmap[u] for u in context_units]
    exp_latent = [h0_gibbs_vmap[u] for u in hidden_units]
    
    # scan requires a function that returns theano expressions, so we cannot pass vmaps in or out. annoying.
    def gibbs_hvh(*args):
        h0_gibbs_vmap = dict(zip(hidden_units, args))
        
        v1_in_vmap = h0_gibbs_vmap.copy()
        v1_in_vmap.update(context_vmap) # add context
        
        v1_activation_vmap = dict((v, v.activation(v1_in_vmap)) for v in visible_units)
        v1_stats_vmap, v1_gibbs_vmap = gibbs_step(rbm, v1_in_vmap, visible_units, mean_field_for_stats, mean_field_for_gibbs)

        h1_in_vmap = v1_gibbs_vmap.copy()
        h1_in_vmap.update(context_vmap) # add context

        h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in hidden_units)
        h1_stats_vmap, h1_gibbs_vmap = gibbs_step(rbm, h1_in_vmap, hidden_units, mean_field_for_stats, mean_field_for_gibbs)
            
        # get the v1 values in a fixed order
        v1_activation_values = [v1_activation_vmap[u] for u in visible_units]
        v1_gibbs_values = [v1_gibbs_vmap[u] for u in visible_units]
        v1_stats_values = [v1_stats_vmap[u] for u in visible_units]
        
        # same for the h1 values
        h1_activation_values = [h1_activation_vmap[u] for u in hidden_units]
        h1_gibbs_values = [h1_gibbs_vmap[u] for u in hidden_units]
        h1_stats_values = [h1_stats_vmap[u] for u in hidden_units]
        
        return v1_activation_values + v1_stats_values + v1_gibbs_values + \
               h1_activation_values + h1_stats_values + h1_gibbs_values
    
    
    # support for persistent CD
    if persistent_vmap is None:
        chain_start = exp_latent
    else:
        chain_start = [persistent_vmap[u] for u in hidden_units]
    
    
    # The 'outputs_info' keyword argument of scan configures how the function outputs are mapped to the inputs.
    # in this case, we want the h1_gibbs_vmap values to map onto the function arguments, so they become
    # h0_gibbs_vmap values in the next iteration. To this end, we construct outputs_info as follows:
    outputs_info = [None] * (len(exp_input)*3) + [None] * (len(exp_latent)*2) + list(chain_start)
    # 'None' indicates that this output is not used in the next iteration.
    
    exp_output_all_list, theano_updates = theano.scan(gibbs_hvh, outputs_info = outputs_info, n_steps = k)
    # we only need the final outcomes, not intermediary values
    exp_output_list = [out[-1] for out in exp_output_all_list]
            
    # reconstruct vmaps from the exp_output_list.
    n_input, n_latent = len(visible_units), len(hidden_units)
    vk_activation_vmap = dict(zip(visible_units, exp_output_list[0:1*n_input]))
    vk_stats_vmap = dict(zip(visible_units, exp_output_list[1*n_input:2*n_input]))
    vk_gibbs_vmap = dict(zip(visible_units, exp_output_list[2*n_input:3*n_input]))
    hk_activation_vmap = dict(zip(hidden_units, exp_output_list[3*n_input:3*n_input+1*n_latent]))
    hk_stats_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+1*n_latent:3*n_input+2*n_latent]))
    hk_gibbs_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+2*n_latent:3*n_input+3*n_latent]))
    
    # add the Theano updates for the persistent CD states:
    if persistent_vmap is not None:
        for u, v in persistent_vmap.items():
            theano_updates[v] = hk_gibbs_vmap[u] # this should be the gibbs vmap, and not the stats vmap!
    
    activation_data_vmap = v0_vmap.copy() # TODO: this doesn't really make sense to have in an activation vmap!
    activation_data_vmap.update(h0_activation_vmap)
    activation_model_vmap = vk_activation_vmap.copy()
    activation_model_vmap.update(context_vmap)
    activation_model_vmap.update(hk_activation_vmap)
    
    stats = Stats(theano_updates) # create a new stats object
    
    # store the computed stats in a dictionary of vmaps.
    stats_data_vmap = v0_vmap.copy()
    stats_data_vmap.update(h0_stats_vmap)
    stats_model_vmap = vk_stats_vmap.copy()
    stats_model_vmap.update(context_vmap)
    stats_model_vmap.update(hk_stats_vmap)
    stats.update({
      'data': stats_data_vmap,
      'model': stats_model_vmap,
    })
            
    stats['data_activation'] = activation_data_vmap
    stats['model_activation'] = activation_model_vmap
        
    return stats

