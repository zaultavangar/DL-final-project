import numpy as np
# from scipy.linalg import orthogonal_procrustes as op
from sklearn.metrics.pairwise import linear_kernel

years = np.arange(2003, 2022, 1)

# Adapted from debiasing lab
def find_similar(vector_matrix, input_vector, num_examples): 
    """
    Use a basic similarity calculation (cosine similarity) to find
    the closest words to an input vector.
    """
    # compute cosine similarity of input_vector with everything else in our vocabulary 
    cosine_similarities = linear_kernel(input_vector, vector_matrix).flatten()
    cosine_similarities /= np.linalg.norm(vector_matrix, axis=1)

    # sort by cosine similarities, to get the most similar vectors on top
    related_words_indices = [i for i in cosine_similarities.argsort()[::-1]]
    return [index for index in related_words_indices][:num_examples]


def find_similar_words(word_of_interest, year, num_examples=10):
    year_file = f'vectors/{year}_vectors.tsv'
    embedding_mat = np.loadtxt(year_file)

    year_meta_file = f'vectors/{year}_metadata.tsv'
    word_mat = np.loadtxt(year_meta_file, dtype=str)

    index_of_interest = np.where(word_mat == word_of_interest)[0][0]
    print(index_of_interest)

    similar_indices = find_similar(embedding_mat, [embedding_mat[index_of_interest]], num_examples)

    print("Words similar to", word_mat[index_of_interest], "in", year)
    for i in similar_indices:
        print(' -', word_mat[i])

find_similar_words("police", 2019)


