#
# Simple example of sentiment classification using the IMDB data set in Keras.
# All fairly standard; trivial to extend binary classification to >2 classes.
#
# Notes:
#
# 1. ALWAYS ensure numpy arrays used with e.g. model.predict() etc; DON'T pass in
# "raw" Python lists - they will silently fail, and output bad results.
#
# E.g., even if "data" is a numpy.ndarray from some known-good data:
#
# stuff = [ [idx for idx in idx_set] for idx_set in data ]
# model.predict(stuff) <= BOOM!
#
# Instead, make sure you're passing in a numpy array:
#
# stuff = np.array( [idx for idx in idx_set] for idx_set in data )
# model.predict(stuff) <= OK!
#
# 2. IMDB dataset in Keras is broken in 1.13/14 by np 1.16.3
#
# 3. Install TensorFlow/Keras nightlies: pip3 install tf_nightly
#

import sys, os, random

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

#
# Some basic parameters for the model/training
#

# Input data
max_vocab  = 10_000 # max. tokens in model "vocabulary"
pad_to     = 256    # input token sequences padded to this length

# Model training
embed_len  = 4      # per-token embedding vector size; 4 to 16 ok for us
N_validate = 10_000 # number of phrases for model validation
N_epochs   = 40     # number of training epochs
batch_size = 512    # larger = potential GPU performance increases

# HDF5 file path to save/load model parameters
model_path = 'model.hdf5'

#
# Special tokens for use in training etc; we're assuming these won't appear in
# the training set texts. Padding should always have value 0.
#

PADDING = ('<PAD>', 0)
START   = ('<START>', 1)
UNKNOWN = ('<UNKNOWN>', 2)
UNUSED  = ('<UNUSED>', 3) # reserved by Keras?

#
# Train using the standard Keras IMDB dataset: reviews that contain both the
# raw text of the review and a simple binary label (0 for negative sentiment,
# 1 for positive).
#
# We'll need a mapping of tokens onto integer ids, and vice versa. IMDB data
# set tokens are lower case, with token numbering UNIT BASED not ZERO BASED.
#
# Be careful - imdb.get_word_index().items() returns ALL words in the IMDB
# data set, even though the data sets returned by imdb.load_data(num_words=N)
# only contain the top N words. If we only add the top N words to tok_to_idx,
# we safely use the model with other texts provided "unknown" words are
# replaced with the UNKNOWN tag specified previously.
#
# Remember to sanitise training data (e.g., puncutation).
#

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
	num_words=max_vocab,
	start_char=START[1],
	oov_char=UNKNOWN[1],
	)

tmp = imdb.get_word_index().items()
tmp = sorted( [(k,v) for k,v in tmp], key=lambda x: x[1] )[:max_vocab] # limit vocab!

tok_to_idx = dict( {x for x in [PADDING,START,UNKNOWN]} )
tok_to_idx.update( {tok:(idx+UNUSED[1]) for tok,idx in tmp} )

idx_to_tok = dict( {(idx,tok) for tok,idx in tok_to_idx.items()} )

#
# Print some info for the user
#

print()
print(f'Training with {len(train_data)} entries ({len(train_labels)} labels')
print(f'Token dictionary has {len(tok_to_idx)} entries')
print()

#
# Define simple, fast model:
#
# - train n-tuple embedding vectors for each distinct word
# - dimensional reduction via GAP, also helps prevent overfitting
# - fully-connected layer (one node per embedding tuple element)
# - binary classifier, so single output with sigmoid activation.
#

model = keras.Sequential()
model.add(keras.layers.Embedding(max_vocab, embed_len))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(embed_len, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(
	optimizer='adam',           # https://arxiv.org/abs/1412.6980v8
	loss='binary_crossentropy', # crossentropy typical for binary classifier
	metrics=['acc']
)

#
# Preprocess training data; need all training phrases with same length
#

pad = lambda x: pad_sequences(x, value=PADDING[1], padding='post', maxlen=pad_to)

train_data = pad(train_data)
test_data = pad(test_data)

#
# Load previously trained model parameters, or train model anew.
#

if os.path.exists(model_path):
	model = keras.models.load_model(model_path)
else:
	train_d = train_data[N_validate:]
	train_l = train_labels[N_validate:]

	valid_d = train_data[:N_validate]
	valid_l = train_labels[:N_validate]

	history = model.fit(train_d, train_l,
		validation_data=(valid_d, valid_l),
		epochs=N_epochs, batch_size=batch_size,
		verbose=1)

	model.save(model_path)

#
# How did we do?
#

results = model.evaluate(test_data, test_labels)

print()
print('Model metrics:')
for i,r in enumerate(results):
	print(f'  {model.metrics_names[i]} : {r}')

#
# Try some random predictions; also demonstrates how to convert between numpy data
# and native python data for the purposes of this model.
#

N_print = 10 # number of random sample phrases to classify
cutoff = 0.5 # negative sentiment: <cutoff

# Random selection of phrases from test data
indices = [ random.randint(0,len(test_data)) for i in range(N_print) ]
phrases = [ test_data[i]   for i in indices ]
labels  = [ test_labels[i] for i in indices ]

# Check we know how to convert data back and forth:
# numpy => list(list(string)) [a] => list(list(int)) [b] => numpy [c]
phrases = [ [idx_to_tok.get(i,UNKNOWN[0]) for i in l] for l in phrases ] # [a]
phrases = [ [tok_to_idx.get(t,UNKNOWN[1]) for t in l] for l in phrases ] # [b]
phrases = np.array( [np.array(x) for x in phrases] ) # [c]

# Predict sentiment of regenerated data.
sentiments = model.predict( phrases )

print()
print('Sample predictions:')
for i in range(len(phrases) ):
	tmp = [idx for idx in phrases[i] if idx != PADDING[1]]
	print()
	print(' '.join( [idx_to_tok.get(idx,UNKNOWN[0]) for idx in tmp] ))
	s = sentiments[i]
	pl, al = (0 if s<cutoff else 1), labels[i]
	print(f'Sentiment: {s}, Predicted label: {pl}, Actual label: {al}')
