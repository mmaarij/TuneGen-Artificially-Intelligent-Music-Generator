#!/usr/bin/env python
# coding: utf-8

# In[4]:


from midi_to_dataframe import NoteMapper, MidiReader, MidiWriter
import IPython
from IPython.display import Image, IFrame
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import music21
import pickle
import nltk
nltk.download('punkt')

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[5]:


note_mapping_config_path = "./config/map-to-group.json"
note_mapper = NoteMapper(note_mapping_config_path)
reader = MidiReader(note_mapper)


# In[13]:


# load the tokenizer object from the saved file
with open("./tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
    
# load the tokenizer object from the saved file
with open("./vocabularyMappings.pkl", 'rb') as f:
    vocabMappings = pickle.load(f)


# In[7]:


checkpoint_path = "./model_checkpoint.h5"
loaded_model = keras.models.load_model(checkpoint_path)


# In[9]:


#Creates word to idx map using tokenizer.word_index
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


# In[8]:


def next_tokens(input_str, n):
    print ("Seed -",  input_str, sep = '\n\n')
    final_string = ''
    for i in range(n):
        token = tokenizer.texts_to_sequences([input_str])[0]
        prediction = loaded_model.predict(token, verbose=0)
        final_string = final_string + reverse_word_map[numpy.argmax(prediction[0])] + ' ' 
        input_str = input_str + ' ' + reverse_word_map[numpy.argmax(prediction[0])]
        input_str = ' '.join(input_str.split(' ')[1:])
    return final_string


# In[22]:


# getting midi files
filepath = "./datasets/extra/"
filename = "SinceUBeenGone.midi"
fullFilePath = filepath+filename

MidiDataDF = pd.DataFrame()
tempDF = reader.convert_to_dataframe(fullFilePath)
MidiDataDF = MidiDataDF.append(tempDF)


# In[23]:


NotesDataDF = MidiDataDF[["notes"]]


# In[24]:


notesTemp = list(NotesDataDF["notes"])
mappedNotes = []

for i in notesTemp:
    indexSplit = i.split(",")
    for j in indexSplit:
        if len(indexSplit) > 1:
            mapping = int(vocabMappings[j]) * (-1)
        else:
            mapping = int(vocabMappings[j])
        mappedNotes.append(mapping)

print(len(mappedNotes))
#mappedNotes


# In[28]:


vocabularyChars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] # 10 is minus (-) and 11 is comma (,)
mappedNotesChars = []

for note in mappedNotes:
    temp = str(note)
    tempArr = [*temp]
    if tempArr[0] == "-":
        tempArr[0] = "10"
    tempArr.append("11")
    mappedNotesChars.extend(tempArr)
    
# Convert the list to a string with each element separated by a space
mappedNotesString = " ".join(mappedNotesChars)

print(len(mappedNotesString))
print(mappedNotesString[0:100])


# In[32]:


input_str = mappedNotesString[0:100]

# Uses first 50 tokens from given input_str as input. Since the seq_length is 50, only 50 tokens are taken using the tokenizer.
output = next_tokens(input_str, 10)
print("\nGenerated string -\n\n", output)


# In[ ]:




