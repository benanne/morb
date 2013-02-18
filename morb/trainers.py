from morb.base import Trainer

import theano
import theano.tensor as T

import numpy as np

class MinibatchTrainer(Trainer):
    # use self.rbm, self.umap, self.get_updates(vmap)
    def compile_function(self, initial_vmap, monitors=[], name='func', mb_size=32, train=True, mode=None):
        # setting train=False is useful when compiling a function for evaluation only, i.e. no training.
        # this is interesting when one wants to evaluate training progress on a validation set, for example.
        # then the variables will not be updated, but there might still be updates from scan operations
        # for example, so we still have to fetch them!
        updates = self.get_theano_updates(train) 
        
        # initialise data sets
        data_sets = {}
        for u, v in initial_vmap.items():
            shape = (1,) * v.ndim
            data_sets[u] = theano.shared(value = np.zeros(shape, dtype=theano.config.floatX),
                                          name="dataset for '%s'"  % u.name)
                                          
        index = T.lscalar() # index to a minibatch
        
        # construct givens for the compiled theano function - mapping variables to data
        givens = dict((initial_vmap[u], data_sets[u][index*mb_size:(index+1)*mb_size]) for u in initial_vmap)
            
        TF = theano.function([index], monitors,
            updates = updates, givens = givens, name = name, mode = mode)    
                
        def func(dmap):
            # dmap is a dict that maps unit types on their respective datasets (numeric).
            units_list = dmap.keys()
            data_sizes = [int(np.ceil(dmap[u].shape[0] / float(mb_size))) for u in units_list]
            
            if data_sizes.count(data_sizes[0]) != len(data_sizes): # check if all data sizes are equal
                raise RuntimeError("The sizes of the supplied datasets for the different input units are not equal.")

            data_cast = [dmap[u].astype(theano.config.floatX) for u in units_list]
            
            for i, u in enumerate(units_list):
                data_sets[u].set_value(data_cast[i], borrow=True)
                
            for batch_index in xrange(min(data_sizes)):
                yield TF(batch_index)
                
        return func
                        

