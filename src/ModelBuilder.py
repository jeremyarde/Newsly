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


def build_model():
    num_words = 5000

    # x_train, y_train, x_test, y_test, labels = DataHelper.get_news_bias_data(deep_model=True)
    x_train, x_test, y_train, y_test, labels = DataHelper.get_bbc_news_data()

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_matrix(x_train, 'binary')
    x_test = tokenizer.texts_to_matrix(x_test, 'binary')

    one_hot_encoder = preprocessing.LabelEncoder()
    y_train = one_hot_encoder.fit_transform(y_train)
    y_test = one_hot_encoder.fit_transform(y_test)

    y_train = keras.utils.to_categorical(y_train, len(labels))
    y_test = keras.utils.to_categorical(y_test, len(labels))

    # PICKLE THE DATA
    # DataHelper.pickle_object('y_train', y_train)
    # DataHelper.pickle_object('x_train', x_train)
    # DataHelper.pickle_object('y_test', y_test)
    # DataHelper.pickle_object('x_test', x_test)

    # GRID SEARCHING
    model = Keras(max_words=num_words, num_classes=len(labels))

    # Testing grid search
    param_grid = dict(
        epochs=[10, 30],
        batch_size=[10, 100],
        dense_layers=[
            30,
            60,
            128,
            150,
            200,
            # 250,
            # 350,
            # 500,
            # 1000,
            # 2000
        ],
        optimizer=[
            # 'adam',
            # 'sgd',
            'nadam',
            'adadelta'
        ],
        activation=[
            # 'relu',
            'softmax',
            # 'elu',
            # 'tanh',
            # 'linear'
        ],
        dropout=[
            # 0.0,
            0.5,
            0.75,
            0.90
        ]
    )

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x_train, y_train)

    best_model = Keras(**grid_result.best_params_)
    best_model.fit(x_train, y_train)
    best_model.model.evaluate(x_test, y_test)

    print("Done")
