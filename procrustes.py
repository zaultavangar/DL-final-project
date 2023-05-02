import numpy as np
from scipy.linalg import orthogonal_procrustes as op

years = np.arange(2003, 2022, 1)

def get_rotations(embedding_dim=128, vocab_size=9999):
    W_ref = np.loadtxt('vectors/2021_vectors.tsv').T # shape after transpose = (embedding_dim, vocab_size)

    W_all = np.zeros((len(years), W_ref.shape[0], W_ref.shape[1])) # shape = (years, embedding_dim, vocab_size)

    for i, year in year:
        year_file = f'vectors/{year}_vectors.tsv'
        year_mat = np.loadtxt(year_file).T # shape after transpose = (embedding_dim, vocab_size)
        R = op(year_mat, W_ref)
        W_all[i] = np.matmul(year_mat, R)

    return W_all.T # shape after transpose = (years, vocab_size, embedding_dim)
