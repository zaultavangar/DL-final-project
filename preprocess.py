import pandas as pd
import tensorflow as tf
import numpy as np
import re
import string
import tqdm

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size. 
# TAKEN DIRECTLY FROM WORD2VEC DOCS

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

df_headlines = pd.read_csv('data/abcnews-date-text.csv') # load csv into pandas DataFrame
print(df_headlines.head(5))
print()

df_headlines['publish_date'] = df_headlines['publish_date'].astype(str) # change type of publish_date column to extract year 

df_headlines['year'] = df_headlines['publish_date'].str[:4] # extract year from publish_date column
print(df_headlines.head(5))
print()

# Tokenize headlines using tf.keras.preprocessing.text.Tokenizer 
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_headlines['headline_text'])
max_length = max([len(sequence) for sequence in tokenizer.texts_to_sequences(df_headlines['headline_text'].values)])

print(f"The maximum headline length is {max_length} words.") # MAX = 15 words

word_counts = tokenizer.word_counts

# Compute the average length of headlines (to help with defining the sequence length later)
num_headlines = len(df_headlines)
num_words = sum(word_counts.values())
avg_length = num_words/num_headlines

print(f'Average headline length: {avg_length:.2f} words') # AVG = 6.56 words
print()

headlines_by_year = df_headlines.groupby('year')['headline_text'].apply(list) # group headlines by year
# for each year, we have a list of headlines 
print(headlines_by_year.head(5))
print()

# print the first 10 headlines 
for year, headlines in headlines_by_year.items():
    print(f'Year: {year}')
    for index, headline in enumerate(headlines):
        if index < 5:
          print(headline)
    print()

# Loop through headlines and write them to a txt file 
  # will have a txt file for each year named year_headlines (e.g. 2003_headlines)

for year, headlines in headlines_by_year.items():
    filename = f'data/{year}_headlines.txt'
    with open(filename, 'w') as f:
        for headline in headlines:
            f.write(headline + '\n')


text_ds_dict = {}

for year in headlines_by_year.keys():
    filename = f'data/{year}_headlines.txt' # get the corresponding file name based on the year
    # Use the non empty lines to construct a tf.data.TextLineDataset object for the next steps
    text_ds = tf.data.TextLineDataset(filename).filter(lambda x: tf.cast(tf.strings.length(x), bool)) # filters out empty lines from dataset
    text_ds_dict[f'{year}'] = text_ds # add the text dataset to the dict

# test the function above works correctly
# text_ds_2003 = text_ds_dict['2003_text_ds']
# for i, line in enumerate(text_ds_2003):
#     if i < 5:
#       print(line)

# Custom standardization function to lowercase the text and remove punctuation
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000 # larger vocab size the better, if not, then too many UNKs in my opinion 
sequence_length = 15 # Using max headline length

vectorize_layer_dict = {}

for year in headlines_by_year.keys():
    # vectorize_layer for each 
    vectorize_layer_dict[f'{year}'] = tf.keras.layers.TextVectorization(
        standardize = custom_standardization,
        max_tokens = vocab_size,
        output_mode = 'int',
        output_sequence_length=sequence_length,
      )

text_vector_ds_dict = {}
sequences_dict = {}

# Loop through each TextLineDataset and TextVectorization layer and adapt the layer to the dataset
for year, text_ds, vectorize_layer in zip(text_ds_dict.keys(), text_ds_dict.values(), vectorize_layer_dict.values()):
    vectorize_layer.adapt(text_ds.batch(1024)) # BATCH SIZE ?

    # Print vocab for each year sorted (descending by frequency)
    inverse_vocab = vectorize_layer.get_vocabulary()
    print('20 most common words in: ')
    print(f'{year}: ', inverse_vocab[:20])
    print()

    # Vectorize the data in each text_ds (not too sure what this does but I'm following the tf docs)
    text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    text_vector_ds_dict[f'{year}'] = text_vector_ds

    # We now have a tf.data.Dataset of integer encoded sentences 
    # To prepare the dataset for training a word2vec model, flatten them into a list of sentence vector sequences 
       # To iterate over each sentence in data set during training

    sequences = list(text_vector_ds.as_numpy_iterator())
    sequences_dict[f'{year}'] = sequences
    # print(len(sequences))
    print("Few examples from sequences: ")
    for seq in sequences[:5]:
      print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
    print()

# Call generate_training_data() to generate training examples for the word2vec model
# The function iterates over each word from each sequence to collect positive and negative context words

# Loop through sequences for each year 
for year, sequences in sequences_dict.items():
  targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=42) # SEED = 42?

  targets = np.array(targets)
  contexts = np.array(contexts)
  labels = np.array(labels)

  print('')
  print(f"targets.shape: {targets.shape}")
  print(f"contexts.shape: {contexts.shape}")
  print(f"labels.shape: {labels.shape}")







   
