import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from adjustText import adjust_text
from sklearn.metrics.pairwise import cosine_similarity

years = list(range(2003, 2022))

def find_word_frequencies():
    dfs = []

    for year in years:
        #print(year)
        df = pd.read_csv(f'vectors/{year}_metadata.tsv', sep='\t', header=None, names=['word'])
        dfs.append(df)

    df_concat = pd.concat(dfs)

    word_counts = df_concat['word'].value_counts()

    # convert Series to DataFrame with two columns
    df_counts = word_counts.reset_index()
    df_counts.columns = ['word', 'count']

    df_filtered = df_counts[df_counts['count'] == 19]['word']
    df_filtered.to_csv('words_with_count_19.tsv', sep='\t', index=False)
    count_19 = df_filtered.count()

    print('Number of words which appear 19 times: ', count_19)
    print(df_counts.head(10))

    return df_counts

    # 1777 words appear 19 times, 
    # Let's look at some of the most frequent AND interesting words from each year, using the year_metadata.tsv file, and see if they appear 19 times
      # If they do, they are candidates for plotting/further analysis

    # Words: police, govt, iraq/iraqi, fire, water, court, death, australia, war, attack, drug, election, trade

def find_word_frequency(word_of_interest):
    df_counts = find_word_frequencies()
    return (df_counts[df_counts['word'] == word_of_interest]['count'].values[0])

def plot_nearest_neighbors(word_of_interest, year, num_examples):


    print(year)
    words_to_indices = embedding_neighbors.find_similar_words(word_of_interest, year, num_examples) # get dictionary of words to indices
    if words_to_indices is None:
        print(f'{word_of_interest} not found for {year}')
        return
    # words_to_indices.update(get_neighbors_of_neighbors(words_to_indices, 5))

    print(words_to_indices)

    labels=[]
    tokens=[]

    year_file = f'vectors/{year}_vectors.tsv'
    embedding_mat = np.loadtxt(year_file)

    year_meta_file = f'vectors/{year}_metadata.tsv'
    word_mat = np.loadtxt(year_meta_file, dtype=str)

    for word in word_mat:
        index_of_interest = np.where(word_mat == word)[0][0]
        tokens.append(embedding_mat[index_of_interest])
        labels.append(word)

    # for rs in range(1, 51):
    tsne_model = TSNE(n_components=2, random_state=9) #30 is good
    reduced_vectors = tsne_model.fit_transform(np.array(tokens))

    specific_indices = np.array(list(words_to_indices.values()))
    reduced_vectors = reduced_vectors[specific_indices, :]
    relevant_words = [labels[index] for index in specific_indices]

    plt.figure(figsize=(15, 8))

    # Assign a color index to each point based on its distance from the point of interest
    distances = np.linalg.norm(reduced_vectors - reduced_vectors[0], axis=1)
    max_distance = np.max(distances)
    color_index = distances / max_distance
    plt.scatter(reduced_vectors[1:, 0], reduced_vectors[1:, 1], 
                c=color_index[1:], cmap='viridis', marker='o', s=50)
    plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1], 
                c='r', marker='*', s=100)
    
    texts = []

    for i, label in enumerate(relevant_words):
        texts.append(plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], label))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black')) # adjust labels so that they're read 

    plt.title(f'Nearest of Neighbors for {word_of_interest}, {year}')
    plt.show()

def get_vector_matrices(year, word1, word2):
    procrustes_mat = np.load('vectors/embeddings_rotated.npy') # shape: years, vocab_size, embedding_size
    rotated_embeddings = np.transpose(procrustes_mat, (2, 0, 1))

    year_meta_file = f'vectors/{year}_metadata.tsv'
    word_mat = np.loadtxt(year_meta_file, dtype=str)

    index1 = np.where(word_mat == word1)[0]
    index2 = np.where(word_mat == word2)[0]

    if index1.size>0 and index2.size>0:
        index1 = index1[0] # get index of interest for word1
        index2 = index2[0] # get index of interest for word2
    else:
        print('One of these words is not available')
        return None, None
    
    vector_word1 = rotated_embeddings[int(year)-2003][index1]
    vector_word2 = rotated_embeddings[int(year)-2003][index2]

    return vector_word1, vector_word2

def get_cosine_similarity(year, word1, word2):

    vector_word1, vector_word2 = get_vector_matrices(year, word1, word2)

    if vector_word1 is None or vector_word2 is None:
        return None
    else:
        cos_sim = cosine_similarity(vector_word1.reshape(1,-1), vector_word2.reshape(1, -1)) # convert to shape 1, 256 and final cos_sim
        return cos_sim[0][0]



def plot_cos_similarities(word1, word2):
    freq_word1 = find_word_frequency(word1)
    freq_word2 = find_word_frequency(word2)
    print()
    print(freq_word1, freq_word2)
    # if find_word_frequency(word1) == find_word_frequency(word2) == 19:
    cos_sim_list = []
    for year in years:
        cos_sim = get_cosine_similarity(year, word1, word2)
        cos_sim_list.append(cos_sim)

    # plot year vs cosine similarity as a scatter plot
    plt.figure(figsize=(15, 8))
    plt.scatter(years, cos_sim_list)
    plt.xlabel('Year')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity between {word1} and {word2}')
    plt.xticks(years)
    plt.grid(True) # show grid on both axes
    plt.ylim(-0.5, 0.5)  # set y-axis limits

    # exclude None values from the scatter plot
    x = [years[i] for i in range(len(cos_sim_list)) if cos_sim_list[i] is not None]
    y = [cos_sim_list[i] for i in range(len(cos_sim_list)) if cos_sim_list[i] is not None]
    plt.scatter(x, y)

    print(len(x))
     # plot dashed lines between the points
    for i in range(len(x)-1):
        if not cos_sim_list[i+1] is None:
            plt.plot([x[i], x[i+1]], [y[i], y[i+1]], '--', color='grey')

    plt.show()

    # else:
    #     print(f'{word1} and {word2} do not appear in all the years from 2003-2021')


find_word_frequencies() # getting word frequencies

# set parameters of interest
word_of_interest = 'epstein'
year = '2020'
num_examples = 10

# find word freq 
word_freq = find_word_frequency(word_of_interest)
print(f'{word_of_interest} word frequency: {word_freq}')

# # plot nearest neighbors
# plot_nearest_neighbors(word_of_interest, year, num_examples)

# plot cosine similarity
plot_cos_similarities('violence', 'terrorist') 
# vaccine, protest; president, election; president, approval; china, trade/export

# plot cosine similarity 


# 1, 4, 5, 9, 17, 
# 21, 30

# 31-50

# def get_neighbors_of_neighbors(words_to_indices, num_examples):
#     new_words_to_indices = {}
#     for word in words_to_indices:
#         neighboring_words_to_indices = embedding_neighbors.find_similar_words(word, year, num_examples=num_examples)
#         for w, i in neighboring_words_to_indices.items():
#             if w not in words_to_indices:
#                 new_words_to_indices[w] = i
#     return new_words_to_indices


# for word, index in words_to_indices.items():
#     if word not in tokens_set:
#         tokens_set.add(word)
#         tokens.append(embedding_mat[index])
#         labels.append(word)
#     neighboring_words_to_indices = embedding_neighbors.find_similar_words(word, year, num_examples=10) # find 10 neighbors of neighbor
#     for w2, i2 in neighboring_words_to_indices.items():
#         if w2 not in tokens_set:
#             tokens_set.add(w2)
#             tokens.append(embedding_mat[i2])
#             labels.append(w2)

# embedding_mat = procrustes_matrix[year-2003]
# #embedding_mat_2d = pca.fit_transform(embedding_mat)
# embedding_mat_2d = tsne.fit_transform(embedding_mat)

# print(embedding_mat_2d.shape)

# year_meta_file = f'vectors/{year}_metadata.tsv'
# word_mat = np.loadtxt(year_meta_file, dtype=str)



# #word_embedding = embedding_mat[index_of_interest] # embedding for the word of interest

# print(embedding_mat_2d[index_of_interest])
# embeddings_2d.append(embedding_mat_2d[index_of_interest])

# print(embeddings_2d)

# for i, year in enumerate(years):
#     plt.scatter(embeddings_2d[i][0], embeddings_2d[i][1], label=year)
# plt.legend()
# plt.show()