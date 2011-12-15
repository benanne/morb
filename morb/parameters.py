from morb.base import Parameters

import theano.tensor as T
from theano.tensor.nnet import conv

        
class ProdParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.W = W
        self.variables = [self.W]
        self.vu = units_list[0]
        self.hu = units_list[1]
        
        self.terms[self.vu] = lambda vmap: T.dot(vmap[self.hu], W.T)
        self.terms[self.hu] = lambda vmap: T.dot(vmap[self.vu], W)
                
    def gradient(self, vmap):
        return [T.dot(vmap[self.vu].T, vmap[self.hu])]
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.hu](vmap) * vmap[self.hu])
        # return - T.sum(T.dot(vmap[self.vu], self.W) * vmap[self.hu])
        # T.sum sums both over the minibatch dimension and the hiddens dimension.
        
    
class BiasParameters(Parameters):
    def __init__(self, rbm, units, b, name=None):
        super(BiasParameters, self).__init__(rbm, [units], name=name)
        self.b = b
        self.variables = [self.b]
        self.u = units
        
        self.terms[self.u] = lambda vmap: self.b
        
    def gradient(self, vmap):
        return [T.sum(vmap[self.u], axis=0)] # sum over axis 0
        # this requires the Units instance to be represented by a matrix variable, i.e. a minibatch

    def energy_term(self, vmap):
        return - T.sum(T.dot(vmap[self.u], self.b)) # T.sum is for minibatches 
        # bias is NOT TRANSPOSED because it's a vector, and apparently vectors are COLUMN vectors by default.


class AdvancedProdParameters(Parameters):
    def __init__(self, rbm, units_list, dimensions_list, W, name=None):
        super(AdvancedProdParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.W = W
        self.variables = [self.W]
        self.vu = units_list[0]
        self.hu = units_list[1]
        self.vd = dimensions_list[0]
        self.hd = dimensions_list[1]
        self.Wd = self.vd + self.hd
        
        # there are vd visible dimensions and hd hidden dimensions, meaning that the weight matrix has
        # vd + hd = Wd dimensions.
        # the hiddens and visibles have hd+1 and vd+1 dimensions respectively, because the first dimension
        # is reserved for minibatches!
        self.terms[self.vu] = lambda vmap: T.tensordot(vmap[self.hu], W, axes=(range(1,self.hd+1),range(self.vd, self.Wd)))
        self.terms[self.hu] = lambda vmap: T.tensordot(vmap[self.vu], W, axes=(range(1,self.vd+1),range(0, self.vd)))
        
    def gradient(self, vmap):
        return [T.tensordot(vmap[self.vu], vmap[self.hu], axes=([0],[0]))] # only sum out the minibatch dimension.
        
    def energy_term(self, vmap):
        # v_part = T.tensordot(vmap[self.vu], self.W, axes=(range(1, self.vd+1), range(0, self.vd)))
        v_part = self.terms[self.hu](vmap)
        neg_energy = T.tensordot(v_part, vmap[self.hu], axes=(range(0, self.hd+1), range(0, self.hd+1)))
        # in this case, we also sum over the minibatches in the 2nd step, hence the ranges are hd+1 long.
        return - neg_energy # don't forget to flip the sign!


class AdvancedBiasParameters(Parameters):
    def __init__(self, rbm, units, dimensions, b, name=None):
        super(AdvancedBiasParameters, self).__init__(rbm, [units], name=name)
        self.b = b
        self.variables = [self.b]
        self.u = units
        self.ud = dimensions
        
        self.terms[self.u] = lambda vmap: self.b
        
    def gradient(self, vmap):
        return [T.sum(vmap[self.u], axis=0)] # sum over minibatch axis
        
    def energy_term(self, vmap):
        return - T.sum(T.tensordot(vmap[self.u], self.b, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        

class SharedBiasParameters(Parameters):
    """
    like AdvancedBiasParameters, but a given number of trailing dimensions are 'shared'.
    """
    def __init__(self, rbm, units, dimensions, shared_dimensions, b, name=None):
        super(SharedBiasParameters, self).__init__(rbm, [units], name=name)
        self.b = b
        self.variables = [self.b]
        self.u = units
        self.ud = dimensions
        self.sd = shared_dimensions
        self.nd = self.ud - self.sd
        
        self.terms[self.u] = lambda vmap: T.shape_padright(self.b, self.sd)
        
    def _shared_axes(self, vmap):
        d = vmap[self.u].ndim
        return range(d - self.sd, d)
        
    def gradient(self, vmap):
        axes = self._shared_axes(vmap)
        return [T.sum(T.mean(vmap[self.u], axis=axes), axis=0)]
        
    def energy_term(self, vmap):
        # b_padded = T.shape_padright(self.b, self.sd)
        # return - T.sum(T.tensordot(vmap[self.u], b_padded, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        # this does not work because tensordot cannot handle broadcastable dimensions.
        # instead, the dimensions of b_padded which are broadcastable should be summed out afterwards.
        # this comes down to the same thing. so:
        t = T.tensordot(vmap[self.u], self.b, axes=(range(1, self.nd+1), range(0, self.nd)))
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
        self.W = W # (hidden_maps, visible_maps, filter_height, filter_width)
        self.variables = [self.W]
        self.vu = units_list[0] # (mb_size, visible_maps, visible_height, visible_width)
        self.hu = units_list[1] # (mb_size, hidden_maps, hidden_height, hidden_width)
        self.shape_info = shape_info

        # conv input is (output_maps, input_maps, filter height [numrows], filter width [numcolumns])
        # conv input is (mb_size, input_maps, input height [numrows], input width [numcolumns])
        # conv output is (mb_size, output_maps, output height [numrows], output width [numcolumns])
        
        def term_vu(vmap):
            # input = hiddens, output = visibles so we need to swap dimensions
            W_shuffled = self.W.dimshuffle(1, 0, 2, 3)
            if self.filter_shape is not None:
                shuffled_filter_shape = [self.filter_shape[k] for k in (1, 0, 2, 3)]
            else:
                shuffled_filter_shape = None
            return conv.conv2d(vmap[self.hu], W_shuffled, border_mode='full', \
                               image_shape=self.hidden_shape, filter_shape=shuffled_filter_shape)
            
        def term_hu(vmap):
            # input = visibles, output = hiddens, flip filters
            W_flipped = self.W[:, :, ::-1, ::-1]
            return conv.conv2d(vmap[self.vu], W_flipped, border_mode='valid', \
                               image_shape=self.visible_shape, filter_shape=self.filter_shape)
        
        self.terms[self.vu] = term_vu
        self.terms[self.hu] = term_hu
    
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
                    
    def gradient(self, vmap):
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
        return [c.dimshuffle(1, 0, 2, 3)]
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.hu](vmap) * vmap[self.hu])
        
        
        
        
# TODO: 1D convolution + optimisation




class ThirdOrderParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None):
        super(ThirdOrderParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 3
        self.W = W
        self.variables = [self.W]
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
                
    def gradient(self, vmap):
        p = vmap[self.u0].dimshuffle(0, 1, 'x') * vmap[self.u1].dimshuffle(0, 'x', 1) # (mb, u0, u1)
        p2 = p.dimshuffle(0, 1, 2, 'x') * vmap[self.u2].dimshuffle(0, 'x', 'x', 1) # (mb, u0, u1, u2)
        return [T.sum(p2, axis=0)] # sum out minibatch dimension
        
    def energy_term(self, vmap):
        return - T.sum(self.terms[self.u1](vmap) * vmap[self.u1])
        # sum is over the minibatch and the u1 dimension.




# TODO: Beta?
class BetaParameters(Parameters):
    def __init__(self, rbm, units_list, W1, W2, U1, U2, name=None):
        super(BetaParameters, self).__init__(rbm, units_list, name=name)
        assert len(units_list) == 2
        self.W1, self.W2, self.U1, self.U2 = W1, W2, U1, U2
        self.variables = [W1, W2, U1, U2]
        vu, hu = units_list
        
        self.terms[vu] = lambda vmap: (T.dot(vmap[hu], W1.T) + T.dot(1 - vmap[hu], W2.T), T.dot(vmap[hu], U1.T) + T.dot(1 - vmap[hu], U2.T)) # dit zijn alfa en beta
        self.terms[hu] = lambda vmap: T.dot(W1 - W2, T.log(vmap[vu])) + T.dot(U1 - U2, T.log(1 - vmap[vu])) # dit gaat door de sigmoid
        
    def gradient(self, vmap):
        pass # TODO LATER: this update has 4 components, for W1, W2, U1 and U2!
        
    def energy_term(self, vmap):
        # TODO
        pass
