import numpy as np
import matplotlib.pyplot as plt


def generate_data(N):
    """Creates a noisy dataset with some simple pattern in it."""
    T = N * 38
    u = np.mat(np.zeros((T, 20)))
    for i in range(1, T, 38):
        if i % 76 == 1:
            u[i - 1:i + 19, :] = np.eye(20)
            u[i + 18:i + 38, :] = np.eye(20)[np.arange(19, -1, -1)]
            u[i - 1:i + 19, :] += np.eye(20)[np.arange(19, -1, -1)] 
        else:
            u[i - 1:i + 19, 1] = 1
            u[i + 18:i + 38, 8] = 1
    return u

def get_context(u, N=4):
    T, D = u.shape
    x = np.zeros((T, D * N))
    for i in range(N - 1, T):
        dat = u[i - 1, :]
        for j in range(2, N + 1):
            dat = np.concatenate((dat, u[i - j, :]), 1)
        x[i, :] = dat
    return x




def plot_data(d):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((28,28)), interpolation='gaussian')
    plt.draw()




def one_hot(vec, dim=None):
    """
    Convert a column vector with indices (normalised) to a one-hot representation.
    Each row is a one-hot vector corresponding to an element in the original vector.
    """
    length = len(vec)
    
    if dim is None: # default dimension is the maximal dimension needed to represent 'vec'
        dim = np.max(vec) + 1
        
    m = np.tile(np.arange(dim), (length, 1))
    return (vec == m)




def load_mnist():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set
    
    

def most_square_shape(num_blocks, blockshape=(1,1)):
    x, y = blockshape
    num_x = np.ceil(np.sqrt(num_blocks * y / float(x)))
    num_y = np.ceil(num_blocks / num_x)
    return (num_x, num_y)  
    
    
    
def visualise_filters(data, dim=6, posneg=True):
    """
    input: a (dim*dim) x H matrix, which is reshaped into filters
    """
    num_x, num_y = most_square_shape(data.shape[1], (dim, dim))
    
    #pad with zeros so that the number of filters equals num_x * num_y
    padding = np.zeros((dim*dim, num_x*num_y - data.shape[1]))
    data_padded = np.hstack([data, padding])
    
    data_split = data_padded.reshape(dim, dim, num_x, num_y)
    
    data_with_border = np.zeros((dim+1, dim+1, num_x, num_y))
    data_with_border[:dim, :dim, :, :] = data_split
    
    filters = data_with_border.transpose(2,0,3,1).reshape(num_x*(dim+1), num_y*(dim+1))
    
    filters_with_left_border = np.zeros((num_x*(dim+1)+1, num_y*(dim+1)+1))
    filters_with_left_border[1:, 1:] = filters
    
    if posneg:
        m = np.abs(data).max()
        plt.imshow(filters_with_left_border, interpolation='nearest', cmap=plt.cm.RdBu, vmin=-m, vmax=m)
    else:
        plt.imshow(filters_with_left_border, interpolation='nearest', cmap=plt.cm.binary, vmin = data.min(), vmax=data.max())


