import io
import re
import string
import tqdm
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
import pickle
import os

BATCH_SIZE = 1024
BUFFER_SIZE = 10000

# Custom standardization function to lowercase the text and remove punctuation
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
embedding_dim=256
num_ns = 4

# FOR TESTING PURPOSES
# embedding_dims = [16, 32, 64, 128, 256, 512] 
loss_values = []
##

targets = np.loadtxt('data_preprocessed/all_targets.txt')
contexts = np.loadtxt('data_preprocessed/all_contexts.txt')
labels = np.loadtxt('data_preprocessed/all_labels.txt')

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)


if not os.path.exists('./vectors/'):
    os.system('mkdir vectors')

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots


vocab_size = 4096
word2vec = Word2Vec(vocab_size, embedding_dim, num_ns)
word2vec.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

history = word2vec.fit(dataset, epochs=10)
loss = history.history['loss'][-1]
loss_values.append(loss)

print(f' Loss: {loss:.4f}')

from_disk = pickle.load(open(f'data/vectorize_layer_full.pkl', "rb"))
vectorize_layer = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) # dummy for initialization
vectorize_layer.set_weights(from_disk['weights'])

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open(f'vectors/all_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open(f'vectors/all_metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()

# FOR PLOTTING EMBEDDING DIMENSION VS. LOSS

# plt.plot(embedding_dims, loss_values)

# # Set the title and axis labels
# plt.title("Relationship between Embedding Dimension and Loss Value")
# plt.xlabel("Embedding Dimension")
# plt.ylabel("Loss")

# # Show the plot
# plt.show()


