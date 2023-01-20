import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import warnings
import pickle
import os
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
np.random.seed(10)


# print all files in dataset
for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# test gpu
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


# load midis as stream
filepath = "./dataset/maestro_v3/"
#Getting midi files
notes = []
count = 0
for i in os.listdir(filepath):
    if i.endswith(".midi"):
        count += 1
        print(count)
        tr = filepath+i

        # read midi file
        midi = converter.parse(tr)

        # parse file into notes / chords - chords are made of multiple notes
        songs = instrument.partitionByInstrument(midi)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))


pikFile = open('./notes.pickle', 'ab')
pickle.dump(notes, pikFile)
pikFile.close()

pikFile = open('./notes.pickle', 'rb')
notes = pickle.load(pikFile)
pikFile.close()

print(all_midis)