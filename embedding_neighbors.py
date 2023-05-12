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
    #print('input vector', input_vector[0].shape)

    # compute cosine similarity of input_vector with everything else in our vocabulary 
    cosine_similarities = linear_kernel(input_vector, vector_matrix).flatten()
    cosine_similarities /= np.linalg.norm(vector_matrix, axis=1)

    # print('cosine sims shape: ', cosine_similarities.shape)

    # sort by cosine similarities, to get the most similar vectors on top
    related_words_indices = [i for i in cosine_similarities.argsort()[::-1]]

    return [index for index in related_words_indices][:num_examples]


def find_similar_words(word_of_interest, year, num_examples=10):
    procrustes_matrix = np.load('vectors/embeddings_rotated.npy') # shape: years, vocab_size, embedding_size
    #print('procrustes matrix shape: ', procrustes_matrix.shape) # shape is (19, 4096, 256)

    procrustes_matrix = np.transpose(procrustes_matrix, (2, 0, 1))
    #print('procrustes matrix shape: ', procrustes_matrix.shape) # shape is (19, 4096, 256)

    year_file = f'vectors/{year}_vectors.tsv'
    embedding_mat = np.loadtxt(year_file)

    #print('shape embedding_mat: ', embedding_mat.shape)

    year_meta_file = f'vectors/{year}_metadata.tsv'
    word_mat = np.loadtxt(year_meta_file, dtype=str)

    #print('shape word_mat: ', word_mat.shape)

    index_of_interest = np.where(word_mat == word_of_interest)[0]
    if index_of_interest.size>0:
        index_of_interest = index_of_interest[0]
    else:
        return None
    #print(index_of_interest)

    similar_indices = find_similar(embedding_mat, [embedding_mat[index_of_interest]], num_examples)
    embedding_mat_rotated = procrustes_matrix[int(year)-2003]

    # same methodology but using procrustes matrix: should and does yield same outputs
    similar_indices_rotated = find_similar(embedding_mat_rotated, [embedding_mat_rotated[index_of_interest]], num_examples) # rotated

    print()
    words_to_indices = {}
    print("Words similar to", word_mat[index_of_interest], "in", year, )
    for i in similar_indices:
        word = word_mat[i]
        words_to_indices[word] = i
        print(' -', word)
    print()

    # print("Words similar to", word_mat[index_of_interest], "in", year, '(rotated embeddings)')
    # for i in similar_indices_rotated:
    #     print(' -', word_mat[i])

    return words_to_indices

# TEST WORDS
find_similar_words("trump", 2014)
find_similar_words("trump", 2015)
find_similar_words("trump", 2016)
find_similar_words("trump", 2017)
find_similar_words("trump", 2018)
find_similar_words("trump", 2019)
find_similar_words("trump", 2020)
find_similar_words("trump", 2021)


