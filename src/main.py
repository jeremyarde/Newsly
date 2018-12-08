import os
from configparser import ConfigParser

import dill
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk import corpus

from src.DataUtilities import DataHelper
from src.DataUtilities.DataCleaner import DataCleaner
from src.Models import SklearnTest, KerasTest
from src.Models.KerasTest import Keras

config = ConfigParser()
config.read('../config.ini')

# dc = DataCleaner()
# dc.clean(config['PATHS']['DataCsv'], "../Data/removed_nonsense.csv")

num_words = 200

x_train, y_train, x_test, y_test, labels = DataHelper.get_news_bias_data(deep_model=True)

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_matrix(x_train, 'binary')
x_test = tokenizer.texts_to_matrix(x_test, 'binary')
# x_train = tokenizer.sequences_to_matrix(x_train.tolist(), mode='tfidf')
# x_test = tokenizer.sequences_to_matrix(x_test.tolist(), mode='tfidf')

one_hot_encoder = preprocessing.LabelEncoder()
y_train = one_hot_encoder.fit_transform(y_train)
y_test = one_hot_encoder.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train, len(labels))
y_test = keras.utils.to_categorical(y_test, len(labels))

model = Keras(max_words=num_words, num_classes=len(labels))
# model.keras_train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
# SklearnTest.run_sklearn(x_train, y_train, x_test, y_test)


# Testing grid search
param_grid = dict(
    epochs=[30],
    batch_size=[500],
    dense_layers=[
        # 30,
        # 60,
        # 128,
        # 150,
        # 200,
        250,
        350,
        500,
        1000,
        2000
    ],
    optimizer=[
        # 'adam',
        # 'sgd',
        # 'nadam?',
        'adad?elta'
    ],
    activation=[
        # 'relu',
        # 'softmax',
        'elu',
        # 'tanh',
        # 'linear'
    ],
    dropout=[
        # 0.0,
        # 0.5,
        # 0.75,
        0.90
    ]
)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(x_train, y_train)

best_model = Keras(**grid_result.best_params_)
best_model.fit(x_train, y_train)
best_model.model.evaluate(x_test, y_test)

pickled_model = best_model
with open("model_test.dill", "wb") as f:
    dill.dump((best_model), f)
print("Done")
