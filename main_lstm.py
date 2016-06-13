'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import cPickle as pickle
import numpy as np


# load data
pieces = []
with open('chord_progressions.txt', 'r') as fp:
    for line in fp.readlines():
        pieces.append(line.strip().split(';'))

# separate chord in root and quality
chords = [c.split(':') for s in pieces for c in s]
c_root = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
c_qual = np.unique([c[1] for c in chords])
root2idx = dict((c, i) for i, c in enumerate(c_root))
qual2idx = dict((c, i) for i, c in enumerate(c_qual))

# create slices for training the RNN
num_dims = len(c_root) + len(c_qual) + 1
maxlen = 20

sequences = []
nxt_chord = []
for song in pieces:
    c_prog = []
    for i, c in enumerate(song):
        if i % 4 == 0:
            c_prog.append('bar')
        c_prog.append(c.split(':'))
    
    for i in range(0, min(len(c_prog), 80) - maxlen):
        sequences.append(c_prog[i: i + maxlen])
        nxt_chord.append(c_prog[i + maxlen])

print('nb sequences:', len(sequences))

# vectorize the training data
print('Vectorization...')
X = np.zeros((len(sequences), maxlen, num_dims), dtype=np.bool)
y = np.zeros((len(sequences), num_dims), dtype=np.bool)
for i, song in enumerate(sequences):
    for t, chord in enumerate(song):
        if chord == 'bar':
            X[i, t, -1] = 1
        else:
            X[i, t, root2idx[chord[0]]] = 1
            X[i, t, qual2idx[chord[1]]+len(root2idx)] = 1
    
    if nxt_chord[i] == 'bar':
        y[i, -1] = 1
    else:
        y[i, root2idx[nxt_chord[i][0]]] = 1
        y[i, qual2idx[nxt_chord[i][1]]+len(root2idx)] = 1
print('Dim(X):', X.shape)


# build the model: 2 stacked LSTM
Nn = 256
dout = 0.25
print('Build model...')
model = Sequential()
model.add(LSTM(Nn, return_sequences=True, input_shape=(maxlen, num_dims)))
model.add(Dropout(dout))
model.add(LSTM(Nn, return_sequences=True))
model.add(Dropout(dout))
model.add(LSTM(Nn, return_sequences=False))
model.add(Dropout(dout))
model.add(Dense(num_dims))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# train the model
for iteration in range(50):
    print()
    print('-' * 50)
    print('Iteration', iteration+1)
    
    model.fit(X, y, batch_size=512, nb_epoch=1)
    model.save_weights(filepath="lstm_weights_n%i.hdf5" % Nn, overwrite=True)
    
