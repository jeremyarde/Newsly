import os
from configparser import ConfigParser

import xlrd as xlrd
from cloudpickle import cloudpickle
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


config = ConfigParser()
config.read('../config.ini')


def read_csv(path_to_csv: str=''):
    df = pd.read_csv(path_to_csv, encoding='latin1')
    return df


def read_excel(path_to_excel: str):
    df = pd.read_excel(xlrd.open_workbook(path_to_excel), engine='xlrd')
    return df


def ibc_data():
    [lib, con, neutral] = cloudpickle.load(open('ibcData.pkl', 'rb'))
    return lib, con, neutral


def get_data_from_source():
    df = read_csv(config['PATHS']['DataCsv'])
    df = df.dropna()  # drop all rows with nan
    return df


def get_series_from_df(df_data: pd.DataFrame, inputs: dict):
    samples = df_data[inputs.get('inputs')].astype('U').unique()
    labels = df_data[inputs.get('labels')].unique()

    return samples, labels


def print_distribution(train_labels):
    sns.countplot(train_labels)
    plt.xlabel('Label')
    plt.title('Distribution of data')


def get_data(deep_model: bool=False):
    df_data = get_data_from_source()
    # samples, labels = get_series_from_df(df_data, {'inputs': 'BIAS', 'labels': 'CLASS'})

    biases_unique = set(df_data['BIAS'].astype('U').unique())
    biases_unique = biases_unique - {'fake', 'pseudoscience'}
    classes = df_data['CLASS'].unique()

    # filter out the undesired labels:
    df_data = df_data[df_data['BIAS'].isin(biases_unique)]

    # sources = df['SOURCE']
    # titles = df['TITLE']

    # replace newlines ( replace with tokenization later...
    # text_data = df_data['TITLE'].str.replace('\n', '')
    # train_labels = df_data['BIAS'].str.replace('\n', '')

    train_labels = df_data['BIAS'].astype('U')
    text_data = df_data['TITLE'].astype('U')
    text_data = text_data.values.astype('U')

    # tfidf stuff
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(text_data.tolist())

    modes = ['count', 'binary', 'tfidf', 'freq']
    encoded_text_data = tokenizer.texts_to_matrix(text_data.tolist(), mode='freq')
    train_data = encoded_text_data

    print_distribution(train_labels)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2)

    # turn the labels into one hot encoded versions
    one_hot_encoder = preprocessing.LabelEncoder()
    y_train = one_hot_encoder.fit_transform(y_train)
    y_test = one_hot_encoder.fit_transform(y_test)

    if deep_model:
        one_hot_encoder = preprocessing.OneHotEncoder()
        y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

    print(f"Counts:\nTrain: {len(x_train)}, {len(y_train)}\nTest: {len(x_test)}, {len(y_test)}")
    print(f"Shapes:\nTrain: {x_train.shape}, {y_train.shape}\nTest: {x_test.shape}, {y_test.shape}")

    return x_train, x_test, y_train, y_test
