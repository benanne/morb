from morb.base import Stats

import numpy as np

import theano


def cd_stats(rbm, v0_vmap, visible_units, hidden_units, context_units=[], k=1, mean_field_for_visibles=True, mean_field_for_stats=True, persistent_vmap=None):
    # with 'mean_field_for_visibles', we can control whether hiddens are sampled based on visibles samples or visible means in the CD iterations.       
    
    # first we need to get the context, because we will have to merge it into the other vmaps.
    context_vmap = dict((u, v0_vmap[u]) for u in context_units)
    
    # now, complete the vmaps and units lists (add missing proxy units)
    v0_vmap = rbm.complete_vmap(v0_vmap)
    context_vmap = rbm.complete_vmap(context_vmap)
    visible_units = rbm.complete_units_list(visible_units)
    hidden_units = rbm.complete_units_list(hidden_units)
    context_units = rbm.complete_units_list(context_units)

    h0_activation_vmap = dict((h, h.activation(v0_vmap)) for h in hidden_units)
    h0_sample_vmap = rbm.sample(hidden_units, v0_vmap) # without mean field (rbm enforces consistency)
    if mean_field_for_stats:
        h0_sample_cd_vmap = rbm.mean_field(hidden_units, v0_vmap) # with mean field
    else:
        h0_sample_cd_vmap = h0_sample_vmap
    
    # add context
    h0_activation_vmap.update(context_vmap)
    h0_sample_cd_vmap.update(context_vmap)
    h0_sample_vmap.update(context_vmap)
    
    exp_input = [v0_vmap[u] for u in visible_units]
    exp_context = [v0_vmap[u] for u in context_units]
    exp_latent = [h0_sample_vmap[u] for u in hidden_units]
    
    # scan requires a function that returns theano expressions, so we cannot pass vmaps in or out. annoying.
    def gibbs_hvh(*args):
        h0_sample_vmap = dict(zip(hidden_units, args)) # these must be without mf!
        
        v1_in_vmap = h0_sample_vmap.copy()
        v1_in_vmap.update(context_vmap)
        
        v1_activation_vmap = dict((v, v.activation(v1_in_vmap)) for v in visible_units)
        v1_sample_vmap = rbm.sample(visible_units, v1_in_vmap) # without mf
        if mean_field_for_stats or mean_field_for_visibles:
            v1_sample_cd_vmap = rbm.mean_field(visible_units, v1_in_vmap) # with mf
        else:
            v1_sample_cd_vmap = v1_sample_vmap
        
        if mean_field_for_visibles:
            h1_in_vmap = v1_sample_cd_vmap.copy()
            # use the mean field version of the visibles to sample hiddens from visibles
        else:
            h1_in_vmap = v1_sample_vmap.copy() # use the sampled visibles to sample hiddens from visibles
        
        h1_in_vmap.update(context_vmap)

        h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in hidden_units)
        h1_sample_vmap = rbm.sample(hidden_units, h1_in_vmap) # without mf
        if mean_field_for_stats:
            h1_sample_cd_vmap = rbm.mean_field(hidden_units, h1_in_vmap) # with mf
        else:
            h1_sample_cd_vmap = h1_sample_vmap

            
        # get the v1 values in a fixed order
        v1_activation_values = [v1_activation_vmap[u] for u in visible_units]
        v1_sample_cd_values = [v1_sample_cd_vmap[u] for u in visible_units]
        v1_sample_values = [v1_sample_vmap[u] for u in visible_units]
        
        # same for the h1 values
        h1_activation_values = [h1_activation_vmap[u] for u in hidden_units]
        h1_sample_cd_values = [h1_sample_cd_vmap[u] for u in hidden_units]
        h1_sample_values = [h1_sample_vmap[u] for u in hidden_units]
        
        return v1_activation_values + v1_sample_cd_values + v1_sample_values + \
               h1_activation_values + h1_sample_cd_values + h1_sample_values
    
    
    # support for persistent CD
    if persistent_vmap is None:
        chain_start = exp_latent
    else:
        chain_start = [persistent_vmap[u] for u in hidden_units]
    
    
    # The 'outputs_info' keyword argument of scan configures how the function outputs are mapped to the inputs.
    # in this case, we want the h1_sample_vmap values to map onto the function arguments, so they become
    # h0_sample_vmap values in the next iteration. To this end, we construct outputs_info as follows:
    outputs_info = [None] * (len(exp_input)*3) + [None] * (len(exp_latent)*2) + list(chain_start)
    # 'None' indicates that this output is not used in the next iteration.
    # We need the non-cd samples as input! so h1_sample_vmap becomes h)_sample_vmap
    
    exp_output_all_list, theano_updates = theano.scan(gibbs_hvh, outputs_info = outputs_info, n_steps = k)
    # we only need the final outcomes, not intermediary values
    exp_output_list = [out[-1] for out in exp_output_all_list]
            
    # reconstruct vmaps from the exp_output_list.
    n_input, n_latent = len(visible_units), len(hidden_units)
    vk_activation_vmap = dict(zip(visible_units, exp_output_list[0:1*n_input]))
    vk_sample_cd_vmap = dict(zip(visible_units, exp_output_list[1*n_input:2*n_input]))
    vk_sample_vmap = dict(zip(visible_units, exp_output_list[2*n_input:3*n_input]))
    hk_activation_vmap = dict(zip(hidden_units, exp_output_list[3*n_input:3*n_input+1*n_latent]))
    hk_sample_cd_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+1*n_latent:3*n_input+2*n_latent]))
    hk_sample_vmap = dict(zip(hidden_units, exp_output_list[3*n_input+2*n_latent:3*n_input+3*n_latent]))
    
    # add the Theano updates for the persistent CD states:
    if persistent_vmap is not None:
        for u, v in persistent_vmap.items():
            theano_updates[v] = hk_sample_vmap[u]
    
    activation_data_vmap = v0_vmap.copy()
    activation_data_vmap.update(h0_activation_vmap)
    activation_model_vmap = vk_activation_vmap.copy()
    activation_model_vmap.update(context_vmap)
    activation_model_vmap.update(hk_activation_vmap)
    
    stats = Stats(theano_updates) # create a new stats object
    
    # store the computed stats in a dictionary of vmaps.
    if mean_field_for_stats:
        stats_data_vmap = v0_vmap.copy()
        stats_data_vmap.update(h0_sample_cd_vmap)
        stats_model_vmap = vk_sample_cd_vmap.copy()
        stats_model_vmap.update(context_vmap)
        stats_model_vmap.update(hk_sample_cd_vmap)
        stats.update({
          'data': stats_data_vmap,
          'model': stats_model_vmap,
        })
    else:
        stats_data_vmap = v0_vmap.copy()
        stats_data_vmap.update(h0_sample_vmap)
        stats_model_vmap = vk_sample_vmap.copy()
        stats_model_vmap.update(context_vmap)
        stats_model_vmap.update(hk_sample_vmap)
        stats.update({
          'data': stats_data_vmap,
          'model': stats_model_vmap,
        })
        
    stats['data_activation'] = activation_data_vmap
    stats['model_activation'] = activation_model_vmap
        
    return stats

