#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from midi_to_dataframe import NoteMapper, MidiReader, MidiWriter
import IPython
from IPython.display import Image, IFrame
from PIL import Image
import seaborn as sns
import os
import json
import music21
import pickle
import nltk
from savetogit import *
nltk.download('punkt')


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


# In[2]:


note_mapping_config_path = "./config/map-to-group.json"
note_mapper = NoteMapper(note_mapping_config_path)


# In[3]:


reader = MidiReader(note_mapper)

# getting midi files
filepath = "./datasets/dataset_pop/"
MidiDataDF = pd.DataFrame()
count = 0

for filename in os.listdir(filepath):
    if filename.endswith(".midi"):
        
        # create file path
        count += 1
        #print(count, filename, end = " ")
        fullFilePath = filepath+filename

        # read file as dataframe
        tempDF = reader.convert_to_dataframe(fullFilePath)
        #print(tempDF.shape[0])
        MidiDataDF = MidiDataDF.append(tempDF)


# In[4]:


notes = '\n'.join(MidiDataDF['notes'])
notes = notes.replace(',', ' , ')
notes = notes.replace('_', ' _ ')
print(type(notes))
print(notes[0:100])


# In[5]:


corpus = notes.split('\n')
print(len(corpus))
print(type(corpus))
print(corpus[:2])


# In[6]:


tokenizer = Tokenizer(filters='!"$%&()*+-/:;<=>?@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_unique_words = len(tokenizer.word_index) + 1
print(total_unique_words)
print(word_index)


# In[7]:


for line in corpus:
    seqs = tokenizer.texts_to_sequences([line])[0]
    print(seqs)


# In[8]:


input_sequences = []
n_gram_len = 5

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, n_gram_len):
        n_gram_seqs = token_list[:i+1]
        input_sequences.append(n_gram_seqs)
        
print(len(input_sequences))
print(input_sequences[:5])


# In[9]:


max_seq_length = max([len(x) for x in input_sequences])
print(max_seq_length)

input_seqs = np.array(pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre'))
print(input_seqs[:5])


# In[10]:


del input_sequences
del MidiDataDF


# In[11]:


x_values, labels = input_seqs[:, :-1], input_seqs[:, -1]

y_values = tf.keras.utils.to_categorical(labels, num_classes=total_unique_words)

print(x_values[:3])
print(labels[:3])


# In[12]:


del input_seqs


# In[13]:


path = 'glove.6B.100d.txt'

embeddings_index = {}

with open(path) as f:
  for line in f:
    values = line.split()
    word = values[0]
    coeffs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coeffs
    
dict(list(embeddings_index.items())[0:2])


# In[14]:


embeddings_matrix = np.zeros((total_unique_words, 100))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


# In[15]:


K.clear_session()
    
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=total_unique_words, output_dim=100, weights=[embeddings_matrix], input_length=max_seq_length-1, trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(total_unique_words, activation='softmax')
])

# Define the checkpoint
#ACCURACY_THRESHOLD = 0.95
checkpoint_name = "model3_checkpoint.h5"
checkpoint_path = "./" + checkpoint_name

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("\nSaving Model After ", epoch, " to GitHub\n")
        saveToGit(checkpoint_name, epoch, logs.get("accuracy"), logs.get("loss"))
        
        #if(logs.get("accuracy") >= ACCURACY_THRESHOLD):   
            #print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
            #self.model.stop_training = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True)

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')

# Train the model with the checkpoint
history = model.fit(x_values, y_values, epochs=150, validation_split=0.2, verbose=1, batch_size=150, callbacks=[checkpoint, CustomCallback()])