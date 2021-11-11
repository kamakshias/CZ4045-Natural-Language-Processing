# importing required packages
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import glob


# load json data
cwd = os.getcwd()

json_path = glob.glob(os.path.join(cwd, "*.json"))[0]
print(json_path)

try:
    data = pd.read_json("reviewSelected100.json", encoding = "ISO-8859-1", lines = True)
except Exception:
    data = pd.read_json(json_path, encoding = "ISO-8859-1", lines = True)


data['Sentiment'] = [1 if x > 3 else 0 for x in data.stars]
X, y = (data['text'].values, data['Sentiment'].values)

data.head()

tk = Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post')


# create LSTM model for sentiment analysis
vocabulary_size = len(tk.word_counts.keys())+1
max_words = 100
embedding_size = 32

model = Sequential([
    Embedding(vocabulary_size, embedding_size, input_length=max_words),
    LSTM(units=200, activation='tanh',
         use_bias=True, kernel_initializer='glorot_uniform',
         kernel_regularizer=tf.keras.regularizers.L2(1e-5), 
         bias_regularizer=tf.keras.regularizers.L2(1e-5)
         ),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# load model checkpoint
save_folder = os.path.join(cwd, "model_checkpoint")
checkpoint_path = os.path.join(save_folder, "lstm_model_checkpoint_v2.ckpt")
model.load_weights(checkpoint_path)


# evaluate model
print("\n")

scores = model.evaluate(X_pad, y)
accuracy = scores[1] * 100

print(f"\nSentiment Analysis accuracy: {accuracy:.3f}%")