#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[28]:


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
import gc
gc.enable()

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

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

# In[29]:


note_mapping_config_path = "./config/map-to-group.json"
note_mapper = NoteMapper(note_mapping_config_path)


# ## Convert a MIDI file to a DataFrame
# 
# The **MidiReader** object is used to read a MIDI file from disk and convert it to a **DataFrame**. A **NoteMapper** object is passed to the MidiReader upon initialization to handle the MIDI to text conversion of note durations and program names.

# In[30]:


reader = MidiReader(note_mapper)

# getting midi files
filepath = "./datasets/dataset_pop/"
MidiDataDF = pd.DataFrame()
count = 0

for filename in os.listdir(filepath):
    if filename.endswith(".midi"):
        
        # create file path
        count += 1
        print(count, filename, end = " ")
        fullFilePath = filepath+filename

        # read file as dataframe
        tempDF = reader.convert_to_dataframe(fullFilePath)
        print(tempDF.shape[0])
        MidiDataDF = MidiDataDF.append(tempDF)


# In[31]:


MidiDataDF


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

# In[32]:


NotesDataDF = MidiDataDF[["notes"]]
NotesDataDF


# ## Vocabulary Building

# In[33]:


vocabulary = ["rest"] # add rest by default


# ### Instruments

# In[34]:


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

# In[35]:


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

# In[36]:


f = open(note_mapping_config_path)
jsonData = json.load(f)
f.close()

print(jsonData["durations"])


# ### Build Vocabulary List

# In[37]:


for i in instruments:
    for c in chordsList:
        for d in jsonData["durations"][i]:
            word = str(i) + "_" + str(c) + "_" + str(d)
            vocabulary.append(word)
            
for p in percussionInstruments:
    word = "percussion_" + str(p) + "_0.25"
    vocabulary.append(word)
            
print(len(vocabulary))


# ### Create Dictionary to Map Word onto Integers

# In[38]:


vocabMappings = dict(zip(vocabulary, range(0, len(vocabulary))))
print(vocabMappings)


# ### Mapping Data To Integers (Forward Mapping)

# In[39]:


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
mappedNotes


# In[40]:


vocabularyChars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"] # 10 is minus (-) and 11 is comma (,)
mappedNotesChars = []

for note in mappedNotes:
    temp = str(note)
    tempArr = [*temp]
    if tempArr[0] == "-":
        tempArr[0] = "10"
    tempArr.append("11")
    mappedNotesChars.extend(tempArr)
    
print(len(mappedNotesChars))
mappedNotesChars


# ## RNN + LSTM Model

# ### Preparing Data For Input Into Model

# In[42]:

gc.collect()

print("To NP:")
mappedNotesChars = np.array(mappedNotesChars, dtype=float)

features = []
targets = []
featureLength = 1000

for i in range(len(mappedNotesChars)-featureLength):
    #print("Index:", i)
    tempF = mappedNotesChars[i:i+featureLength]
    features.append(tempF)
    tempT = mappedNotesChars[i+featureLength]
    targets.append(tempT)
    
n_patterns = len(targets)


# In[43]:


print(len(targets))
print(len(features[0]))


# In[45]:


print(features[0][0])
print(targets[0])


# In[ ]:

pickle.dump(featureLength, open('pop_featureLength.pickle', 'wb'))
print("Feature Length Pickled")
pickle.dump(targets, open('pop_targets.pickle', 'wb'))
print("Target Pickled")
pickle.dump(features, open('pop_features.pickle', 'wb'))
print("Feature Pickled")