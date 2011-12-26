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


