from morb.base import Parameters

import theano.tensor as T
from theano.tensor.nnet import conv



class FixedBiasParameters(Parameters):
    # Bias fixed at -1, which is useful for some energy functions (like Gaussian with fixed variance, Beta)
    def __init__(self, rbm, units, name=None):
        super(FixedBiasParameters, self).__init__(rbm, [units], name=name)
        self.variables = []
        self.u = units
        
        self.terms[self.u] = lambda vmap: -T.ones_like(vmap[self.u])
        
    def energy_term(self, vmap):
        return T.sum(vmap[self.u]) # NO minus sign! bias is -1 so this is canceled.
        
        
class ProdParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]
        
        self.terms[self.vu] = lambda vmap: T.dot(vmap[self.hu], W.T)
        self.terms[self.hu] = lambda vmap: T.dot(vmap[self.vu], W)
        
        self.energy_gradients[self.var] = lambda vmap: T.dot(vmap[self.vu].T, vmap[self.hu])
                
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.hu](vmap) * vmap[self.hu])
        # return - T.sum(T.dot(vmap[self.vu], self.var) * vmap[self.hu])
        # T.sum sums both over the minibatch dimension and the hiddens dimension.
        
    
class BiasParameters(Parameters):
    def __init__(self, rbm, units, b, name=None):
        super(BiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units
        
        self.terms[self.u] = lambda vmap: self.var
        
        self.energy_gradients[self.var] = lambda vmap: T.sum(vmap[self.u], axis=0) # sum over minibatch axis
        
    def energy_term(self, vmap):
        return - T.sum(T.dot(vmap[self.u], self.var)) # T.sum is for minibatches 
        # bias is NOT TRANSPOSED because it's a vector, and apparently vectors are COLUMN vectors by default.


class AdvancedProdParameters(Parameters):
    def __init__(self, rbm, units_list, dimensions_list, W, name=None):
        super(AdvancedProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]
        self.vd = dimensions_list[0]
        self.hd = dimensions_list[1]
        self.vard = self.vd + self.hd
        
        # there are vd visible dimensions and hd hidden dimensions, meaning that the weight matrix has
        # vd + hd = Wd dimensions.
        # the hiddens and visibles have hd+1 and vd+1 dimensions respectively, because the first dimension
        # is reserved for minibatches!
        self.terms[self.vu] = lambda vmap: T.tensordot(vmap[self.hu], W, axes=(range(1,self.hd+1),range(self.vd, self.vard)))
        self.terms[self.hu] = lambda vmap: T.tensordot(vmap[self.vu], W, axes=(range(1,self.vd+1),range(0, self.vd)))
        
        self.energy_gradients[self.var] = lambda vmap: T.tensordot(vmap[self.vu], vmap[self.hu], axes=([0],[0]))
        # only sums out the minibatch dimension.
                
    def energy_term(self, vmap):
        # v_part = T.tensordot(vmap[self.vu], self.var, axes=(range(1, self.vd+1), range(0, self.vd)))
        v_part = self.terms[self.hu](vmap)
        neg_energy = T.tensordot(v_part, vmap[self.hu], axes=(range(0, self.hd+1), range(0, self.hd+1)))
        # in this case, we also sum over the minibatches in the 2nd step, hence the ranges are hd+1 long.
        return - neg_energy # don't forget to flip the sign!


class AdvancedBiasParameters(Parameters):
    def __init__(self, rbm, units, dimensions, b, name=None):
        super(AdvancedBiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions
        
        self.terms[self.u] = lambda vmap: self.var
        
        self.energy_gradients[self.var] = lambda vmap: T.sum(vmap[self.u], axis=0) # sum over minibatch axis
        
    def energy_term(self, vmap):
        return - T.sum(T.tensordot(vmap[self.u], self.var, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        

class SharedBiasParameters(Parameters):
    """
    like AdvancedBiasParameters, but a given number of trailing dimensions are 'shared'.
    """
    def __init__(self, rbm, units, dimensions, shared_dimensions, b, name=None):
        super(SharedBiasParameters, self).__init__(rbm, [units], name=name)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions
        self.sd = shared_dimensions
        self.nd = self.ud - self.sd
        
        self.terms[self.u] = lambda vmap: T.shape_padright(self.var, self.sd)
        
        self.energy_gradients[self.var] = lambda vmap: T.sum(T.mean(vmap[self.u], axis=self._shared_axes(vmap)), axis=0)
        
    def _shared_axes(self, vmap):
        d = vmap[self.u].ndim
        return range(d - self.sd, d)
            
    def energy_term(self, vmap):
        # b_padded = T.shape_padright(self.var, self.sd)
        # return - T.sum(T.tensordot(vmap[self.u], b_padded, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        # this does not work because tensordot cannot handle broadcastable dimensions.
        # instead, the dimensions of b_padded which are broadcastable should be summed out afterwards.
        # this comes down to the same thing. so:
        t = T.tensordot(vmap[self.u], self.var, axes=(range(1, self.nd+1), range(0, self.nd)))
        # now sum t over its trailing shared dimensions, which mimics broadcast + tensordot behaviour.
        axes = range(t.ndim - self.sd, t.ndim)
        t2 = T.sum(t, axis=axes)
        # finally, sum out minibatch axis
        return - T.sum(t2, axis=0)
        

               
class Convolutional2DParameters(Parameters):
    def __init__(self, rbm, units_list, W, shape_info=None, name=None):
        # use the shape_info parameter to provide a dict with keys:
        # hidden_maps, visible_maps, filter_height, filter_width, visible_height, visible_width, mb_size
        
        super(Convolutional2DParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.var = W # (hidden_maps, visible_maps, filter_height, filter_width)
        self.variables = [self.var]
        self.vu = units_list[0] # (mb_size, visible_maps, visible_height, visible_width)
        self.hu = units_list[1] # (mb_size, hidden_maps, hidden_height, hidden_width)
        self.shape_info = shape_info

        # conv input is (output_maps, input_maps, filter height [numrows], filter width [numcolumns])
        # conv input is (mb_size, input_maps, input height [numrows], input width [numcolumns])
        # conv output is (mb_size, output_maps, output height [numrows], output width [numcolumns])
        
        def term_vu(vmap):
            # input = hiddens, output = visibles so we need to swap dimensions
            W_shuffled = self.var.dimshuffle(1, 0, 2, 3)
            if self.filter_shape is not None:
                shuffled_filter_shape = [self.filter_shape[k] for k in (1, 0, 2, 3)]
            else:
                shuffled_filter_shape = None
            return conv.conv2d(vmap[self.hu], W_shuffled, border_mode='full', \
                               image_shape=self.hidden_shape, filter_shape=shuffled_filter_shape)
            
        def term_hu(vmap):
            # input = visibles, output = hiddens, flip filters
            W_flipped = self.var[:, :, ::-1, ::-1]
            return conv.conv2d(vmap[self.vu], W_flipped, border_mode='valid', \
                               image_shape=self.visible_shape, filter_shape=self.filter_shape)
        
        self.terms[self.vu] = term_vu
        self.terms[self.hu] = term_hu
        
        def gradient(vmap):
            if self.visible_shape is not None:
                i_shape = [self.visible_shape[k] for k in [1, 0, 2, 3]]
            else:
                i_shape = None
        
            if self.hidden_shape is not None:
                f_shape = [self.hidden_shape[k] for k in [1, 0, 2, 3]]
            else:
                f_shape = None
            
            v_shuffled = vmap[self.vu].dimshuffle(1, 0, 2, 3)
            h_shuffled = vmap[self.hu].dimshuffle(1, 0, 2, 3)
            
            c = conv.conv2d(v_shuffled, h_shuffled, border_mode='valid', image_shape=i_shape, filter_shape=f_shape)   
            return c.dimshuffle(1, 0, 2, 3)
            
        self.energy_gradients[self.var] = gradient
    
    @property    
    def filter_shape(self):
        keys = ['hidden_maps', 'visible_maps', 'filter_height', 'filter_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def visible_shape(self):
        keys = ['mb_size', 'visible_maps', 'visible_height', 'visible_width']                
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def hidden_shape(self):
        keys = ['mb_size', 'hidden_maps', 'visible_height', 'visible_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            hidden_height = self.shape_info['visible_height'] - self.shape_info['filter_height'] + 1
            hidden_width = self.shape_info['visible_width'] - self.shape_info['filter_width'] + 1
            return (self.shape_info['mb_size'], self.shape_info['hidden_maps'], hidden_height, hidden_width)
        else:
            return None
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.hu](vmap) * vmap[self.hu])
        
        
        
        
# TODO: 1D convolution + optimisation




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
            p = T.tensordot(vmap[self.u1], W, axes=([1],[1])) # (mb, u0, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u0)
            # cannot use two tensordots here because of the minibatch dimension.
            
        def term_u1(vmap):
            p = T.tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u1)
            
        def term_u2(vmap):
            p = T.tensordot(vmap[self.u0], W, axes=([1],[0])) # (mb, u1, u2)
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




class ThirdOrderFactoredParameters(Parameters):
    """
    Factored 3rd order parameters, connecting three Units instances. Each factored
    parameter matrix has dimensions (units_size, num_factors).
    """
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
