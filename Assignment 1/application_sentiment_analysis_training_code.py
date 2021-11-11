# importing required packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

data = pd.read_json('reviewSelected100.json', encoding = "ISO-8859-1", lines = True)

data['Sentiment'] = [1 if x > 3 else 0 for x in data.stars]
X, y = (data['text'].values, data['Sentiment'].values)

data.head()

tk = Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

# Splitting the dataset into train (70%), test (20%), and val (10%)
X_train_, X_test, y_train_, y_test = train_test_split(X_pad, y, test_size = 0.2, random_state = 1)
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size = 0.125, random_state = 1)

original_data_size = len(X_pad)
train_size = len(X_train)
test_size = len(X_test)
val_size = len(X_val)

if not (train_size / original_data_size == 0.7 
    and test_size / original_data_size == 0.2
    and val_size / original_data_size == 0.1):
    raise RuntimeError("Train:Test:Val split is not 70:20:10")

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

num_epochs = 50

model.fit(X_train, y_train,
          validation_data=(X_val,y_val),
          batch_size=32,
          epochs=num_epochs)

scores = model.evaluate(X_test,y_test,verbose=0)
print("\nTest accuracy:",scores[1])

cwd = os.getcwd()
save_folder = os.path.join(cwd, "model_checkpoint")
os.mkdir(save_folder)
checkpoint_path = os.path.join(save_folder, "lstm_model_checkpoint_v2.ckpt")
model.save_weights(checkpoint_path)

