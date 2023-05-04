import numpy as np
from scipy.linalg import orthogonal_procrustes as op

years = np.arange(2003, 2022, 1)

def get_rotations(embedding_dim=128, vocab_size=9999):
    W_ref = np.loadtxt('vectors/all_vectors.tsv').T[:,:vocab_size] # shape after transpose = (embedding_dim, vocab_size)

    W_all = np.zeros((len(years), W_ref.shape[0], W_ref.shape[1])) # shape = (years, embedding_dim, vocab_size)

    for i, year in enumerate(years):
        year_file = f'vectors/{year}_vectors.tsv'
        year_mat = np.loadtxt(year_file).T[:,:vocab_size] # shape after transpose = (embedding_dim, vocab_size)
        R = op(year_mat, W_ref)
        print(year_mat.shape)
        W_all[i] = np.matmul(year_mat, R[0])
        print(i, " done")

    return W_all.T # shape after transpose = (years, vocab_size, embedding_dim)

get_rotations(vocab_size=4096)
print("Done!")
