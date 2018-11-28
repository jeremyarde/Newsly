import os
from configparser import ConfigParser

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import corpus

from src.DataUtilities import DataHelper
from src.Models import SklearnTest, KerasTest

config = ConfigParser()
config.read('../config.ini')

#
# stemmer = Stemmer('english')
# tokenizer = Tokenizer('english')
#
# lsa = SumySummarizer(summarizerType=SummarizerType.LSA)
# ed = SumySummarizer(summarizerType=SummarizerType.Edmundson)
# lex = SumySummarizer(summarizerType=SummarizerType.LexRank)
# rand = SumySummarizer(summarizerType=SummarizerType.Random)
#
# url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
# url2 = "https://www.bbc.com/news/business-45986510"
#
# results = {'lsa': lsa.get_summary(url2),
#            'ed': ed.get_summary(url2),
#            'lex': lex.get_summary(url2),
#            'rand': rand.get_summary(url2)}
#
# print(results)


x_train, y_train, x_test, y_test = DataHelper.get_data()

# KerasTest.keras_train(x_train, y_train, x_test, y_test)

# predict_texts = ['trying to predict on a sentence', 'other sentence to try and predict on']
# predict_texts = tokenizer.texts_to_matrix(predict_texts, mode='count')
SklearnTest.run_sklearn(x_train, y_train, x_test, y_test)


print("Done")
