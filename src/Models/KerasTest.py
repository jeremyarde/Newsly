from configparser import ConfigParser

import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from src.DataUtilities import DataHelper

config = ConfigParser()
config.read('../config.ini')


class Keras:
    def __init__(self, num_classes, max_words):
        self.tokenizer = Tokenizer(num_words=max_words)
        self.num_classes = num_classes

        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(max_words,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.metrics_names)

    def keras_train(self, x_train, y_train, x_test, y_test):
        sns.countplot(y_train)

        batch_size = 32
        epochs = 3

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_split=0.1)
        score = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def get_data_ready(self, x_train, y_train, x_test, y_test):
        x_train = self.tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = self.tokenizer.sequences_to_matrix(x_test, mode='binary')

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
