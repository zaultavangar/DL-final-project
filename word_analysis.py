import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def find_word_frequencies():
    dfs = []
    years = list(range(2003, 2022))

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

    # 1777 words appear 19 times, 
    # Let's look at some of the most frequent AND interesting words from each year, using the year_metadata.tsv file, and see if they appear 19 times
      # If they do, they are candidates for plotting/further analysis

    # Words: police, govt, iraq/iraqi, fire, water, court, death, australia, war, attack, drug, election, trade

    print(df_counts[df_counts['word'] == 'us']['count'].values[0])



def plot_word_movement():
    procrustes_matrix = np.load('vectors/embeddings_rotated.npy') # shape: years, vocab_size_embedding_size
    procrustes_matrix = np.transpose(procrustes_matrix, (2, 0, 1)) # convert shape to 19, 4096, 256

    years = list(range(2003, 2022))


    pca = PCA(n_components=2)

    embeddings_2d = []

    # for year in years:

    labels=[]
    tokens=[]

    # year_meta_file = f'vectors/{year}_metadata.tsv'

    embedding_mat = procrustes_matrix[0] # i.e. 2003
    year_meta_file = f'vectors/2003_metadata.tsv'
    word_mat = np.loadtxt(year_meta_file, dtype=str)

    for word in word_mat:
        index_of_interest = np.where(word_mat == word)[0][0]
        tokens.append(embedding_mat[index_of_interest])
        labels.append(word)

    print(np.array(tokens).shape)
    print(np.array(labels).shape)

    tsne_model = TSNE(n_components=2, perplexity=30, init='pca', n_iter=2500, random_state=42)
    new_values = tsne_model.fit_transform(np.array(tokens))

    x = []
    y =[]
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], 
          xy=(x[i], y[i]),
          xytext=(5,2),
          textcoords='offset points',
          ha = 'right',
          va = 'bottom')
        
    plt.show()



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

      
find_word_frequencies()
plot_word_movement()

