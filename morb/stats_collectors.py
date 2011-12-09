from morb.base import StatsCollector

import numpy as np

import theano

class CDkStatsCollector(StatsCollector):
    def __init__(self, rbm, input_units, latent_units, context_units=[], k=1, mean_field_for_visibles=True, mean_field_for_stats=True):
        super(CDkStatsCollector, self).__init__()
        self.rbm = rbm
        self.input_units = input_units # the units that are supplied to the statscollector as input (i.e. the visibles)
        self.latent_units = latent_units
        self.context_units = context_units
        self.k = k               
        self.mean_field_for_visibles = mean_field_for_visibles
        # with 'mean_field_for_visibles', we can control whether hiddens are sampled based on visibles samples or visible means in the CD iterations.
        # This requires that the Units instances have samplers that return means when cd=True.
        # disabling this option is useful when one doesn't want to apply mean field during the gibbs sampling,
        # but the statistics should be mean field nevertheless to improve convergence.
        self.mean_field_for_stats = mean_field_for_stats
        # 'mean_field_for_stats' controls whether the returned statistics are means or samples. You will almost
        # always want to leave this enabled, as mean stats improve convergence. Note that a sampler that responds
        # to cd=True is required, else this does nothing.
        
    def calculate_stats(self, v0_vmap):
        # first we need to get the context, because we will have to merge it into the other vmaps.
        context_vmap = dict((u, v0_vmap[u]) for u in self.context_units)
    
        h0_linear_activation_vmap = dict((h, h.linear_activation(v0_vmap)) for h in self.latent_units)
        h0_activation_vmap = dict((h, h.activation(v0_vmap)) for h in self.latent_units)
        h0_sample_cd_vmap = dict((h, h.sample(v0_vmap, cd=True)) for h in self.latent_units) # with mean field
        h0_sample_vmap = dict((h, h.sample(v0_vmap)) for h in self.latent_units) # without mean field
        
        # add context
        h0_linear_activation_vmap.update(context_vmap)
        h0_activation_vmap.update(context_vmap)
        h0_sample_cd_vmap.update(context_vmap)
        h0_sample_vmap.update(context_vmap)
        
        exp_input = [v0_vmap[u] for u in self.input_units]
        exp_context = [v0_vmap[u] for u in self.context_units]
        exp_latent = [h0_sample_vmap[u] for u in self.latent_units]
        
        # scan requires a function that returns theano expressions, so we cannot pass vmaps in or out. annoying.
        def gibbs_hvh(*args):
            h0_sample_vmap = dict(zip(self.latent_units, args)) # these must be without mf!
            
            v1_in_vmap = h0_sample_vmap.copy()
            v1_in_vmap.update(context_vmap)
            
            v1_linear_activation_vmap = dict((v, v.linear_activation(v1_in_vmap)) for v in self.input_units)
            v1_activation_vmap = dict((v, v.activation(v1_in_vmap)) for v in self.input_units)
            v1_sample_cd_vmap = dict((v, v.sample(v1_in_vmap, cd=True)) for v in self.input_units) # with mf
            v1_sample_vmap = dict((v, v.sample(v1_in_vmap)) for v in self.input_units) # without mf
            
            if self.mean_field_for_visibles:
                h1_in_vmap = v1_sample_cd_vmap.copy()
                h1_in_vmap.update(context_vmap)
                
                # use the mean field version of the visibles to sample hiddens from visibles
                h1_linear_activation_vmap = dict((h, h.linear_activation(h1_in_vmap)) for h in self.latent_units)
                h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in self.latent_units)
                h1_sample_cd_vmap = dict((h, h.sample(h1_in_vmap, cd=True)) for h in self.latent_units) # with mf
                h1_sample_vmap = dict((h, h.sample(h1_in_vmap)) for h in self.latent_units) # without mf
            else:
                h1_in_vmap = v1_sample_vmap.copy()
                h1_in_vmap.update(context_vmap)
                
                # use the sampled visibles to sample hiddens from visibles
                h1_linear_activation_vmap = dict((h, h.linear_activation(h1_in_vmap)) for h in self.latent_units)
                h1_activation_vmap = dict((h, h.activation(h1_in_vmap)) for h in self.latent_units)
                h1_sample_cd_vmap = dict((h, h.sample(h1_in_vmap, cd=True)) for h in self.latent_units) # with mf
                h1_sample_vmap = dict((h, h.sample(h1_in_vmap)) for h in self.latent_units) # without mf
                

            # get the v1 values in a fixed order
            v1_linear_activation_values = [v1_linear_activation_vmap[u] for u in self.input_units]
            v1_activation_values = [v1_activation_vmap[u] for u in self.input_units]
            v1_sample_cd_values = [v1_sample_cd_vmap[u] for u in self.input_units]
            v1_sample_values = [v1_sample_vmap[u] for u in self.input_units]
            
            # same for the h1 values
            h1_linear_activation_values = [h1_linear_activation_vmap[u] for u in self.latent_units]
            h1_activation_values = [h1_activation_vmap[u] for u in self.latent_units]
            h1_sample_cd_values = [h1_sample_cd_vmap[u] for u in self.latent_units]
            h1_sample_values = [h1_sample_vmap[u] for u in self.latent_units]
            
            return v1_linear_activation_values + v1_activation_values + v1_sample_cd_values + v1_sample_values + \
                   h1_linear_activation_values + h1_activation_values + h1_sample_cd_values + h1_sample_values
        
        # The 'outputs_info' keyword argument of scan configures how the function outputs are mapped to the inputs.
        # in this case, we want the h1_sample_vmap values to map onto the function arguments, so they become
        # h0_sample_vmap values in the next iteration. To this end, we construct outputs_info as follows:
        outputs_info = [None] * (len(exp_input)*4) + [None] * (len(exp_latent)*3) + list(exp_latent)
        # 'None' indicates that this output is not used in the next iteration.
        # We need the non-cd samples as input! so h1_sample_vmap becomes h)_sample_vmap
        
        exp_output_all_list, updates = theano.scan(gibbs_hvh, outputs_info = outputs_info, n_steps = self.k)
        # we only need the final outcomes, not intermediary values
        exp_output_list = [out[-1] for out in exp_output_all_list]
                
        # reconstruct vmaps from the exp_output_list.
        n_input, n_latent = len(self.input_units), len(self.latent_units)
        vk_linear_activation_vmap = dict(zip(self.input_units, exp_output_list[0:n_input]))
        vk_activation_vmap = dict(zip(self.input_units, exp_output_list[n_input:2*n_input]))
        vk_sample_cd_vmap = dict(zip(self.input_units, exp_output_list[2*n_input:3*n_input]))
        vk_sample_vmap = dict(zip(self.input_units, exp_output_list[3*n_input:4*n_input]))
        hk_linear_activation_vmap = dict(zip(self.latent_units, exp_output_list[4*n_input:4*n_input+n_latent]))
        hk_activation_vmap = dict(zip(self.latent_units, exp_output_list[4*n_input+n_latent:4*n_input+2*n_latent]))
        hk_sample_cd_vmap = dict(zip(self.latent_units, exp_output_list[4*n_input+2*n_latent:4*n_input+3*n_latent]))
        hk_sample_vmap = dict(zip(self.latent_units, exp_output_list[4*n_input+3*n_latent:4*n_input+4*n_latent]))
                
        # TODO: some of these are not used... maybe they'll come in handy later? If not, remove them.
        
        self.theano_updates = updates
        
        linear_activation_data_vmap = v0_vmap.copy()
        linear_activation_data_vmap.update(h0_linear_activation_vmap)
        linear_activation_model_vmap = vk_linear_activation_vmap.copy()
        linear_activation_model_vmap.update(context_vmap)
        linear_activation_model_vmap.update(hk_linear_activation_vmap)
        
        activation_data_vmap = v0_vmap.copy()
        activation_data_vmap.update(h0_activation_vmap)
        activation_model_vmap = vk_activation_vmap.copy()
        activation_model_vmap.update(context_vmap)
        activation_model_vmap.update(hk_activation_vmap)
        
        
        # store the computed stats in a dictionary of vmaps.
        if self.mean_field_for_stats:
            stats_data_vmap = v0_vmap.copy()
            stats_data_vmap.update(h0_sample_cd_vmap)
            stats_model_vmap = vk_sample_cd_vmap.copy()
            stats_model_vmap.update(context_vmap)
            stats_model_vmap.update(hk_sample_cd_vmap)
            self.stats = {
              'data': stats_data_vmap,
              'model': stats_model_vmap,
            }
        else:
            stats_data_vmap = v0_vmap.copy()
            stats_data_vmap.update(h0_sample_vmap)
            stats_model_vmap = vk_sample_vmap.copy()
            stats_model_vmap.update(context_vmap)
            stats_model_vmap.update(hk_sample_vmap)
            self.stats = {
              'data': stats_data_vmap,
              'model': stats_model_vmap,
            }
            
        self.stats['data_linear_activation'] = linear_activation_data_vmap
        self.stats['model_linear_activation'] = linear_activation_model_vmap
        self.stats['data_activation'] = activation_data_vmap
        self.stats['model_activation'] = activation_model_vmap
            
        # TODO: add activation and linear activation to the stats as well, both positive and negative
        

        

class PCDStatsCollector(StatsCollector):
    def calculate_stats(self):
        pass # access self.rbm #TODO later
        

