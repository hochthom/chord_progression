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
import numpy as np
import random



def sample2D(a, n, temperature=1.0):
    ri = sample(a[:n]/np.sum(a[:n]), temperature)
    qi = sample(a[n:-1]/np.sum(a[n:-1]), temperature)
    return (ri, qi)
    
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def prettify(seq):
    txt = []
    nb = 0
    for c in seq:
        if c == '|':
            nb += 1
            txt.append(c)
            if nb>1 and (nb-1) % 4 == 0:
                txt.append('\n|')
        else:
            txt.append('%7s' % (c[0]+c[1]))
    #txt.append('|')
    return ' '.join(txt)

    

# load data
pieces = []
with open('chord_progressions.txt', 'r') as fp:
    for line in fp.readlines():
        pieces.append(line.strip().split(';'))

# separate chord in root and quality
chords = [c.split(':') for s in pieces for c in s]
c_root = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
c_qual = np.unique([c[1] for c in chords])
root2idx = dict((c, i) for i, c in enumerate(c_root))
qual2idx = dict((c, i) for i, c in enumerate(c_qual))
idx2root = dict((i, c) for i, c in enumerate(c_root))
idx2qual = dict((i, c) for i, c in enumerate(c_qual))


# create slices for training the RNN
num_dims = len(c_root) + len(c_qual) + 1
maxlen = 20

sequences = []
for song in pieces:
    c_prog = []
    for i, c in enumerate(song):
        if i % 4 == 0:
            c_prog.append('|')
        c_prog.append(c.split(':'))
    
    sequences.append(c_prog)

print('nb sequences:', len(sequences))


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
model.load_weights(filepath="lstm_weights_n256.hdf5")


print('Generating sequences ...')
with open('gen_sequences.txt', 'wb') as fout:
    for j in range(2):
        # choose initial chord seq
        sample_index = random.randint(0, len(sequences))
        song = sequences[sample_index]
        
        X = np.zeros((1, maxlen, num_dims), dtype=np.bool)
        for t, chord in enumerate(song[:maxlen]):
            if chord == '|':
                X[0, t, -1] = 1
            else:
                X[0, t, root2idx[chord[0]]] = 1
                X[0, t, qual2idx[chord[1]]+len(root2idx)] = 1
        
        res = {'root':c_root, 'qual':c_qual, 'sample_idx':sample_index, 'samples':{}}
        for diversity in [0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            gen_seq = [s for s in song[:maxlen]]
            
            x = X.copy()
                
            for i in xrange(48):
                if len(gen_seq) % 5 == 0:
                    gen_seq.append('|')
                    x[0,:-1,:] = x[0,1:,:]
                    x[0,-1,:] = 0
                    x[0,-1,-1] = 1
                    
                preds = model.predict(x, verbose=0)[0]
                nxt_chord = sample2D(preds, len(root2idx), diversity)
                
                gen_seq.append([idx2root[nxt_chord[0]], idx2qual[nxt_chord[1]]])
    
                x[0,:-1,:] = x[0,1:,:]
                x[0,-1,:] = 0
                x[0,-1,nxt_chord[0]] = 1
                x[0,-1,nxt_chord[1]+len(root2idx)] = 1
                    
            print(prettify(gen_seq))
            res['samples'][diversity] = gen_seq
    
        # write to file
        fout.write('\n%2i: sequence %i\n' % (j, sample_index))
        for d in sorted(res['samples'].keys()):
            seq = res['samples'][d]
            fout.write('Diversity: %.1f\n' % d)
            fout.write(prettify(seq))
            fout.write('\n---\n')
    
#        with open('tmp/sample_%i.pkl' % sample_index, 'wb') as fp:
#            pickle.dump(res, fp, -1)
    
    
