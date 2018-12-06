from configparser import ConfigParser

import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
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

#
def create_model(num_classes: int = 5, max_words: int = 100):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.metrics_names)
    return model


class Keras(KerasClassifier):
    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.dense_layers, input_shape=(self.max_words,)))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        print(self.model.metrics_names)

    def __init__(self, batch_size: int = 100, epochs: int = 10, num_classes: int = 5, max_words: int = 100,
                 activation: str = 'relu', dropout: float = 0.5, optimizer: str = 'adam', dense_layers: int = 512,
                 build_fn=None, **kwargs):
        self.deep_model = True
        self.dense_layers = dense_layers
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout
        self.num_classes = num_classes
        self.max_words = max_words
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = create_model()
        super().__init__(build_fn, **kwargs)

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, verbose=1, validation_split=0.1)

    def score(self, x_test, y_test, **kwargs):
        score = self.model.evaluate(x_test, y_test, verbose=1)
        # 0 = test loss
        # 1 = accuracy
        return score[1]

    def get_params(self, deep: bool=True):
        return dict(num_classes=self.num_classes,
                    max_words=self.max_words,
                    activation=self.activation,
                    dropout=self.dropout,
                    optimizer=self.optimizer,
                    dense_layers=self.dense_layers,
                    )

    def set_params(self, **kwargs):
        self.optimizer = kwargs['optimizer']
        self.activation = kwargs['activation']
        self.dropout = kwargs['dropout']
        self.dense_layers = kwargs['dense_layers']
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']

    def keras_train(self, x_train, y_train, x_test, y_test):
        # sns.countplot(y_train)

        batch_size = 100
        epochs = 30

        history = self.model.fit(x_train, y_train, batch_size=batch_size,
                                 epochs=epochs, verbose=1, validation_split=0.1)
        score = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def get_data_ready(self, x_train, y_train, x_test, y_test):
        x_train = self.tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = self.tokenizer.sequences_to_matrix(x_test, mode='binary')

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
