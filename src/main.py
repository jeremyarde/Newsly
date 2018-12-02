import os
from configparser import ConfigParser
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import corpus

from src.DataUtilities import DataHelper
from src.DataUtilities.DataCleaner import DataCleaner
from src.Models import SklearnTest, KerasTest
from src.Models.KerasTest import Keras

config = ConfigParser()
config.read('../config.ini')

# dc = DataCleaner()
# dc.clean(config['PATHS']['DataCsv'], "../Data/removed_nonsense.csv")


x_train, y_train, x_test, y_test = DataHelper.get_news_bias_data(deep_model=True)


# model = Keras(max_words=100, )
# model.keras_train(x_train, y_train, x_test, y_test)
# SklearnTest.run_sklearn(x_train, y_train, x_test, y_test)


print("Done")
