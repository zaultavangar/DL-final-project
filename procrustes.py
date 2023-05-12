import numpy as np
from scipy.linalg import orthogonal_procrustes as op

embedding_dim = 256

years = np.arange(2003, 2022, 1)

ref_words = np.loadtxt('words_with_count_19.tsv', dtype=str)
unk_pos = np.where(ref_words == '[UNK]')[0][0] # find unk
ref_words = np.concatenate((ref_words[:unk_pos], ref_words[unk_pos+1:]), axis=0) # get rid of unk
nw = len(ref_words)

W_all = np.loadtxt('vectors/all_vectors.tsv') # shape = (words, embedding_dim)
all_words = np.loadtxt('vectors/all_metadata.tsv', dtype=str)
W_ref = np.zeros((nw, embedding_dim))
for i, word in enumerate(ref_words):
    pos = np.where(all_words == word)
    W_ref[i] = W_all[pos]

W_ref = W_ref # shape (nw = 1777, embedding_dim)

def get_rotations(embedding_dim=256, vocab_size=9999):

    W_combined = np.zeros((len(years), vocab_size, W_ref.shape[1])) # shape = (years, vocab_size, embedding_dim)

    for i, year in enumerate(years):
        W_year = np.loadtxt(f'vectors/{year}_vectors.tsv') # shape = (vocab, embedding_dim)
        year_words = np.loadtxt(f'vectors/{year}_metadata.tsv', dtype=str)
        W_year_subset = np.zeros((nw, embedding_dim))
        for j, word in enumerate(ref_words):
            pos = np.where(year_words == word)
            W_year_subset[j] = W_year[pos]
        #W_year_subset = W_year_subset # shape after transpose = (nw = 1777, embedding_dim)

        print("sub-matrix found")

        R = op(W_year_subset, W_ref)[0]
        print('Args: ', W_year.shape, R.shape)
        print(i)
        W_combined[i] = np.matmul(W_year, R) # shape = (vocab, embedding_dim)
        print(i, " done")
        # W_combined.shape = (years, vocab, embedding_dim)

    return np.transpose(W_combined, (1, 2, 0)) # shape after transpose = (vocab, embedding_dim, years)

rotations = get_rotations(vocab_size=4095)
print(rotations.shape)
np.save('vectors/embeddings_rotated.npy', rotations) # Save the rotations to a file
print("Done!")

