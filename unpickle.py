import pandas as pd
import tensorflow as tf
import numpy as np
import re
import string
import tqdm
import pickle
# from preprocess import custom_standardization

# Custom standardization function to lowercase the text and remove punctuation
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')

from_disk = pickle.load(open('data/2003_vectorize_layer.pkl', "rb"))
new_v = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk['weights'])

# Lets see the Vector for word "this"
print (new_v("this"))
print (new_v("aussie"))
print (new_v("this aussie ate a burger"))