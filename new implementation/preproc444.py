#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[36]:


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


# # Load Mappings File
# 
# A **NoteMapper** object encapsulates how MIDI note information is converted to text to be displayed within a DataFrame. This object is initialized from a JSON file, containing three objects:
# 
# * **midi-to-text**: JSON mapping of MIDI program numbers to their textual representation. Used when converting MIDI files to DataFrames.
#     * For example: *{"0": "piano"}*
# * **text-to-midi**: JSON mapping of textual representations of MIDI instruments to MIDI program numbers. Used when writing DataFrames to MIDI.
#     * For example: *{"piano": 0}*
# * **durations**: JSON mapping of textual representations of MIDI instruments to predefined quantization values (in quarter notes). Used when converted MIDI files to DataFrames.
#     * For example: *{"piano": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3, 4, 6, 8, 12, 16]}*

# In[37]:


note_mapping_config_path = "./config/map-to-group.json"
note_mapper = NoteMapper(note_mapping_config_path)


# ## Convert a MIDI file to a DataFrame
# 
# The **MidiReader** object is used to read a MIDI file from disk and convert it to a **DataFrame**. A **NoteMapper** object is passed to the MidiReader upon initialization to handle the MIDI to text conversion of note durations and program names.

# In[38]:


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


# In[39]:


#MidiDataDF


# ## MIDI DataFrame
# 
# The created DataFrame object contains the sequence of musical notes found in the input MIDI file, quantized by 16th notes and the following rows:
# 
# * **timestamp**: the MIDI timestamp (tick)
# * **bpm**: the beats per minute at the timestamp
# * **time_signature**: the time signature at the timestamp
# * **measure**: the measure number at the timestamp
# * **beat**: the downbeat within the current measure at the timestamp, in quarter notes
# * **notes**: a textual representation of the notes played at the current timestamp

# In[40]:


NotesDataDF = MidiDataDF[["notes"]]
#NotesDataDF


# ## Vocabulary Building

# In[41]:


vocabulary = ["rest"] # add rest by default


# ### Instruments

# In[42]:


instruments = ['bass', 'synthlead', 'synthfx', 'reed',
               'percussive', 'organ', 'guitar', 'pipe',
               'soundfx', 'chromatic', 'ethnic', 'piano',
               'brass', 'synthpad', 'ensemble', 'strings']

percussionInstruments = ['acousticbassdrum', 'bassdrum', 'rimshot', 'acousticsnare',
                         'clap', 'snare', 'lowfloortom', 'closedhat', 'highfloortom',
                         'pedalhat', 'lowtom', 'openhat', 'lowmidtom', 'highmidtom',
                         'crashcymbal', 'hightom', 'ridecymbal', 'chinesecymbal',
                         'ridebell', 'tambourine', 'splashcymbal', 'cowbell', 'vibraslap',
                         'highbongo', 'lowbongo', 'mutehighconga', 'openhighconga', 'lowconga',
                         'hightimbale', 'lowtimbale', 'highagogo', 'lowagogo', 'cabasa',
                         'maracas', 'shortwhistle', 'longwhistle', 'shortguiro', 'longguiro',
                         'claves', 'highwoodblock', 'lowwoodblock', 'mutecuica', 'opencuica',
                         'mutetriangle', 'opentriangle']


# ### Chords

# In[43]:


notesTemp = list(NotesDataDF["notes"])
chordsList = []

for i in notesTemp:
    if i != "rest":
        indexSplit = i.split(",")
        for j in indexSplit:
            chord = j.split("_")
            if chord[0] != "percussion":
                chordsList.append(chord[1])
        
chordsList = set(chordsList)
print(chordsList)


# ### Durations

# In[44]:


f = open(note_mapping_config_path)
jsonData = json.load(f)
f.close()

print(jsonData["durations"])


# ### Build Vocabulary List

# In[45]:


for i in instruments:
    for c in chordsList:
        for d in jsonData["durations"][i]:
            word = str(i) + "_" + str(c) + "_" + str(d)
            vocabulary.append(word)
            
for p in percussionInstruments:
    word = "percussion_" + str(p) + "_0.25"
    vocabulary.append(word)
            
print(len(vocabulary))
#print(vocabulary)


# ### Create Dictionary to Map Word onto Integers

# In[46]:


vocabMappings = dict(zip(vocabulary, range(0, len(vocabulary))))

# Pickle the tokenizer object and save it to a file
with open("./vocabularyMappings.pkl", 'wb') as f:
    pickle.dump(vocabMappings, f)
#print(vocabMappings)


# ### Mapping Data To Integers (Forward Mapping)

# In[32]:


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


# In[33]:


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


# In[34]:


words = nltk.word_tokenize(mappedNotesString)
print("The number of tokens is", len(words)) 

unique_tokens = set(words)
print("The number of unique tokens are", len(unique_tokens)) 
#prints the number of unique tokens


# In[35]:


vocab_size = 12  #chosen based on statistics of the model
oov_tok = '<OOV>'
embedding_dim = 125
padding_type='post'
trunc_type='post'

# tokenizes sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts([mappedNotesString])
word_index = tokenizer.word_index


tokens = tokenizer.texts_to_sequences([mappedNotesString])[0]

# Pickle the tokenizer object and save it to a file
with open("./tokenizer.pkl", 'wb') as f:
    pickle.dump(tokenizer, f)


# In[19]:


dataX = []
dataY = []
seq_length = 75

for i in range(0, len(tokens) - seq_length-1 , 1):
  seq_in = tokens[i:i + seq_length]
  seq_out = tokens[i + seq_length]

  if seq_out==1: #Skip samples where target word is OOV
    continue
    
  dataX.append(seq_in)
  dataY.append(seq_out)
 
N = len(dataX)
print ("Total training data size is -", N)

X = np.array(dataX)

# one hot encodes the output variable
y = np.array(dataY)
y = np_utils.to_categorical(dataY)


# In[ ]:


# Define checkpoint path and filename
checkpoint_path = "./model_checkpoint.h5"

# Create a ModelCheckpoint callback that saves the model weights only when validation accuracy improves
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# In[58]:


model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print the model summary
model.summary()


# In[ ]:


# Train the model with checkpoint
num_epochs = 30
history = model.fit(X, y, epochs=num_epochs, batch_size=128, verbose=1, validation_split=0.2, callbacks=[checkpoint])